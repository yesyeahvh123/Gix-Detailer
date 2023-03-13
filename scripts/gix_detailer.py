import os, sys, gc, json, copy, math, glob, re
from io import BytesIO
from enum import Enum

import gradio as gr
from PIL import Image, ImageDraw, ImageOps
from modules import ui
from modules.ui_components import FormRow, FormGroup, ToolButton, FormHTML
from modules import processing, shared, images, devices, scripts, script_callbacks, modelloader, masking
from modules.processing import StableDiffusionProcessing
from modules.processing import Processed, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img
from modules.shared import opts, state, cmd_opts
import torch

from webui import wrap_gradio_gpu_call

import pandas as pd
import numpy as np

from typing import Tuple, List, Dict

import cv2
from pathlib import Path
from huggingface_hub import hf_hub_download
from modules.deepbooru import re_special as tag_escape_pattern

BASEDIR = Path(scripts.basedir())

def CopyAtoB(a, b):
    for key, value in a.__dict__.items():
        if key in b.__dict__.keys():
            b.__dict__[key] = value

#================================================================================
#S. (External Source) DDetailer =============================================
#https://github.com/dustysys/ddetailer
#================================================================================
from modules.sd_models import model_hash
from modules.paths import models_path
from basicsr.utils.download_util import load_file_from_url

dd_models_path = os.path.join(models_path, "mmdet")

def dd_list_models(model_path):
        model_list = modelloader.load_models(model_path=model_path, ext_filter=[".pth"])
        
        def modeltitle(path, shorthash):
            abspath = os.path.abspath(path)

            if abspath.startswith(model_path):
                name = abspath.replace(model_path, '')
            else:
                name = os.path.basename(path)

            if name.startswith("\\") or name.startswith("/"):
                name = name[1:]

            shortname = os.path.splitext(name.replace("/", "_").replace("\\", "_"))[0]

            return f'{name} [{shorthash}]', shortname
        
        models = []
        for filename in model_list:
            h = model_hash(filename)
            title, short_model_name = modeltitle(filename, h)
            models.append(title)
        
        return models

def dd_startup():
    from launch import is_installed, run
    if not is_installed("mmdet"):
        python = sys.executable
        run(f'"{python}" -m pip install -U openmim', desc="Installing openmim", errdesc="Couldn't install openmim")
        run(f'"{python}" -m mim install mmcv-full', desc=f"Installing mmcv-full", errdesc=f"Couldn't install mmcv-full")
        run(f'"{python}" -m pip install mmdet', desc=f"Installing mmdet", errdesc=f"Couldn't install mmdet")

    if (len(dd_list_models(dd_models_path)) == 0):
        print("No detection models found, downloading...")
        bbox_path = os.path.join(dd_models_path, "bbox")
        segm_path = os.path.join(dd_models_path, "segm")
        load_file_from_url("https://huggingface.co/dustysys/ddetailer/resolve/main/mmdet/bbox/mmdet_anime-face_yolov3.pth", bbox_path)
        load_file_from_url("https://huggingface.co/dustysys/ddetailer/raw/main/mmdet/bbox/mmdet_anime-face_yolov3.py", bbox_path)
        load_file_from_url("https://huggingface.co/dustysys/ddetailer/resolve/main/mmdet/segm/mmdet_dd-person_mask2former.pth", segm_path)
        load_file_from_url("https://huggingface.co/dustysys/ddetailer/raw/main/mmdet/segm/mmdet_dd-person_mask2former.py", segm_path)

dd_startup()

def dd_gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}

def dd_modeldataset(model_shortname):
    path = dd_modelpath(model_shortname)
    if ("mmdet" in path and "segm" in path):
        dataset = 'coco'
    else:
        dataset = 'bbox'
    return dataset

def dd_modelpath(model_shortname):
    model_list = modelloader.load_models(model_path=dd_models_path, ext_filter=[".pth"])
    model_h = model_shortname.split("[")[-1].split("]")[0]
    for path in model_list:
        if ( model_hash(path) == model_h):
            return path

def dd_update_result_masks(results, masks):
    for i in range(len(masks)):
        boolmask = np.array(masks[i], dtype=bool)
        results[2][i] = boolmask
    return results

def dd_create_segmask_preview(results, image):
    labels = results[0]
    bboxes = results[1]
    segms = results[2]

    cv2_image = np.array(image)
    cv2_image = cv2_image[:, :, ::-1].copy()

    for i in range(len(segms)):
        color = np.full_like(cv2_image, np.random.randint(100, 256, (1, 3), dtype=np.uint8))
        alpha = 0.2
        color_image = cv2.addWeighted(cv2_image, alpha, color, 1-alpha, 0)
        cv2_mask = segms[i].astype(np.uint8) * 255
        cv2_mask_bool = np.array(segms[i], dtype=bool)
        centroid = np.mean(np.argwhere(cv2_mask_bool),axis=0)
        centroid_x, centroid_y = int(centroid[1]), int(centroid[0])

        cv2_mask_rgb = cv2.merge((cv2_mask, cv2_mask, cv2_mask))
        cv2_image = np.where(cv2_mask_rgb == 255, color_image, cv2_image)
        text_color = tuple([int(x) for x in ( color[0][0] - 100 )])
        name = labels[i]
        score = bboxes[i][4]
        score = str(score)[:4]
        text = name + ":" + score
        cv2.putText(cv2_image, text, (centroid_x - 30, centroid_y), cv2.FONT_HERSHEY_DUPLEX, 0.4, text_color, 1, cv2.LINE_AA)
    
    if ( len(segms) > 0):
        preview_image = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
    else:
        preview_image = image

    return preview_image

def dd_is_allblack(mask):
    cv2_mask = np.array(mask)
    return cv2.countNonZero(cv2_mask) == 0

def dd_bitwise_and_masks(mask1, mask2):
    cv2_mask1 = np.array(mask1)
    cv2_mask2 = np.array(mask2)
    cv2_mask = cv2.bitwise_and(cv2_mask1, cv2_mask2)
    mask = Image.fromarray(cv2_mask)
    return mask

def dd_subtract_masks(mask1, mask2):
    cv2_mask1 = np.array(mask1)
    cv2_mask2 = np.array(mask2)
    cv2_mask = cv2.subtract(cv2_mask1, cv2_mask2)
    mask = Image.fromarray(cv2_mask)
    return mask

def dd_dilate_masks(masks, dilation_factor, iter=1):
    if dilation_factor == 0:
        return masks
    dilated_masks = []
    kernel = np.ones((dilation_factor,dilation_factor), np.uint8)
    for i in range(len(masks)):
        cv2_mask = np.array(masks[i])
        dilated_mask = cv2.dilate(cv2_mask, kernel, iter)
        dilated_masks.append(Image.fromarray(dilated_mask))
    return dilated_masks

def dd_offset_masks(masks, offset_x, offset_y):
    if (offset_x == 0 and offset_y == 0):
        return masks
    offset_masks = []
    for i in range(len(masks)):
        cv2_mask = np.array(masks[i])
        offset_mask = cv2_mask.copy()
        offset_mask = np.roll(offset_mask, -offset_y, axis=0)
        offset_mask = np.roll(offset_mask, offset_x, axis=1)
        
        offset_masks.append(Image.fromarray(offset_mask))
    return offset_masks

def dd_combine_masks(masks):
    initial_cv2_mask = np.array(masks[0])
    combined_cv2_mask = initial_cv2_mask
    for i in range(1, len(masks)):
        cv2_mask = np.array(masks[i])
        combined_cv2_mask = cv2.bitwise_or(combined_cv2_mask, cv2_mask)
    
    combined_mask = Image.fromarray(combined_cv2_mask)
    return combined_mask


def dd_create_segmasks(results):
    segms = results[2]
    segmasks = []
    for i in range(len(segms)):
        cv2_mask = segms[i].astype(np.uint8) * 255
        mask = Image.fromarray(cv2_mask)
        segmasks.append(mask)

    return segmasks

import mmcv
from mmdet.core import get_classes
from mmdet.apis import (inference_detector, init_detector)

def dd_get_device():
    device_id = shared.cmd_opts.device_id
    if device_id is not None:
        cuda_device = f"cuda:{device_id}"
    else:
        cuda_device = "cpu"
    return cuda_device

def dd_inference(image, modelname, conf_thres, label):
    path = dd_modelpath(modelname)
    if ( "mmdet" in path and "bbox" in path ):
        results = dd_inference_mmdet_bbox(image, modelname, conf_thres, label)
    elif ( "mmdet" in path and "segm" in path):
        results = dd_inference_mmdet_segm(image, modelname, conf_thres, label)
    return results

def dd_inference_mmdet_segm(image, modelname, conf_thres, label):
    model_checkpoint = dd_modelpath(modelname)
    model_config = os.path.splitext(model_checkpoint)[0] + ".py"
    model_device = dd_get_device()
    model = init_detector(model_config, model_checkpoint, device=model_device)
    mmdet_results = inference_detector(model, np.array(image))
    bbox_results, segm_results = mmdet_results
    dataset = dd_modeldataset(modelname)
    classes = get_classes(dataset)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_results)
    ]
    n,m = bbox_results[0].shape
    if (n == 0):
        return [[],[],[]]
    labels = np.concatenate(labels)
    bboxes = np.vstack(bbox_results)
    segms = mmcv.concat_list(segm_results)
    filter_inds = np.where(bboxes[:,-1] > conf_thres)[0]
    results = [[],[],[]]
    for i in filter_inds:
        results[0].append(label + "-" + classes[labels[i]])
        results[1].append(bboxes[i])
        results[2].append(segms[i])

    return results

def dd_inference_mmdet_bbox(image, modelname, conf_thres, label):
    model_checkpoint = dd_modelpath(modelname)
    model_config = os.path.splitext(model_checkpoint)[0] + ".py"
    model_device = dd_get_device()
    model = init_detector(model_config, model_checkpoint, device=model_device)
    results = inference_detector(model, np.array(image))
    cv2_image = np.array(image)
    cv2_image = cv2_image[:, :, ::-1].copy()
    cv2_gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)

    segms = []
    for (x0, y0, x1, y1, conf) in results[0]:
        cv2_mask = np.zeros((cv2_gray.shape), np.uint8)
        cv2.rectangle(cv2_mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
        cv2_mask_bool = cv2_mask.astype(bool)
        segms.append(cv2_mask_bool)
    
    n,m = results[0].shape
    if (n == 0):
        return [[],[],[]]
    bboxes = np.vstack(results[0])
    filter_inds = np.where(bboxes[:,-1] > conf_thres)[0]
    results = [[],[],[]]
    for i in filter_inds:
        results[0].append(label)
        results[1].append(bboxes[i])
        results[2].append(segms[i])

    return results

#================================================================================
#E. (External Source) DDetailer =============================================
#================================================================================


#================================================================================
#S. (External Source) WD Tagger 1.4 =============================================
#https://github.com/toriato/stable-diffusion-webui-wd14-tagger
#================================================================================

def make_square(img, target_size):
    old_size = img.shape[:2]
    desired_size = max(old_size)
    desired_size = max(desired_size, target_size)

    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]
    new_im = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return new_im


def smart_resize(img, size):
    # Assumes the image has already gone through make_square
    if img.shape[0] > size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    elif img.shape[0] < size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return img
#=========================================================================
# select a device to process
use_cpu = ('all' in shared.cmd_opts.use_cpu) or ('interrogate' in shared.cmd_opts.use_cpu)

if use_cpu:
    tf_device_name = '/cpu:0'
else:
    tf_device_name = '/gpu:0'


    if shared.cmd_opts.device_id is not None:
        try:
            tf_device_name = f'/gpu:{int(shared.cmd_opts.device_id)}'
        except ValueError:
            print('--device-id is not a integer')

class Interrogator:
    @staticmethod
    def postprocess_tags(
        tags: Dict[str, float],

        threshold=0.35,
        additional_tags: List[str] = [],
        exclude_tags: List[str] = [],
        sort_by_alphabetical_order=False,
        add_confident_as_weight=False,
        replace_underscore=False,
        replace_underscore_excludes: List[str] = [],
        escape_tag=False
    ) -> Dict[str, float]:
        for t in additional_tags:
            tags[t] = 1.0

        # those lines are totally not "pythonic" but looks better to me
        tags = {
            t: c
            # sort by tag name or confident
            for t, c in sorted( tags.items(), key=lambda i: i[0 if sort_by_alphabetical_order else 1], reverse=not sort_by_alphabetical_order )

            # filter tags
            if ( c >= threshold and t not in exclude_tags and t.replace('_', ' ') not in exclude_tags )
        }

        new_tags = []
        for tag in list(tags):
            new_tag = tag

            if replace_underscore and tag not in replace_underscore_excludes:
                new_tag = new_tag.replace('_', ' ')

            if escape_tag:
                new_tag = tag_escape_pattern.sub(r'\\\1', new_tag)

            if add_confident_as_weight:
                new_tag = f'({new_tag}:{tags[tag]})'

            new_tags.append((new_tag, tags[tag]))
        tags = dict(new_tags)

        return tags

    def __init__(self, name: str) -> None:
        self.name = name

    def load(self):
        raise NotImplementedError()

    def unload(self) -> bool:
        unloaded = False

        if hasattr(self, 'model') and self.model is not None:
            del self.model
            unloaded = True
            print(f'Unloaded {self.name}')

        if hasattr(self, 'tags'):
            del self.tags

        return unloaded

    def interrogate(
        self,
        image: Image
    ) -> Tuple[
        Dict[str, float],  # rating confidents
        Dict[str, float]  # tag confidents
    ]:
        raise NotImplementedError()

class DeepDanbooruInterrogator(Interrogator):
    def __init__(self, name: str, project_path: os.PathLike) -> None:
        super().__init__(name)
        self.project_path = project_path

    def load(self) -> None:
        print(f'Loading {self.name} from {str(self.project_path)}')

        # deepdanbooru package is not include in web-sd anymore
        # https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/c81d440d876dfd2ab3560410f37442ef56fc663
        from launch import is_installed, run_pip
        if not is_installed('deepdanbooru'):
            package = os.environ.get(
                'DEEPDANBOORU_PACKAGE',
                'git+https://github.com/KichangKim/DeepDanbooru.git@d91a2963bf87c6a770d74894667e9ffa9f6de7ff'
            )

            run_pip(
                f'install {package} tensorflow tensorflow-io', 'deepdanbooru')

        import tensorflow as tf

        # tensorflow maps nearly all vram by default, so we limit this
        # https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
        # TODO: only run on the first run
        for device in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(device, True)

        with tf.device(tf_device_name):
            import deepdanbooru.project as ddp

            self.model = ddp.load_model_from_project(
                project_path=self.project_path,
                compile_model=False
            )

            print(f'Loaded {self.name} model from {str(self.project_path)}')

            self.tags = ddp.load_tags_from_project(
                project_path=self.project_path
            )

    def unload(self) -> bool:
        # unloaded = super().unload()

        # if unloaded:
        #     # tensorflow suck
        #     # https://github.com/keras-team/keras/issues/2102
        #     import tensorflow as tf
        #     tf.keras.backend.clear_session()
        #     gc.collect()

        # return unloaded

        # There is a bug in Keras where it is not possible to release a model that has been loaded into memory.
        # Downgrading to keras==2.1.6 may solve the issue, but it may cause compatibility issues with other packages.
        # Using subprocess to create a new process may also solve the problem, but it can be too complex (like Automatic1111 did).
        # It seems that for now, the best option is to keep the model in memory, as most users use the Waifu Diffusion model with onnx.

        return False

    def interrogate(
        self,
        image: Image
    ) -> Tuple[
        Dict[str, float],  # rating confidents
        Dict[str, float]  # tag confidents
    ]:
        # init model
        if not hasattr(self, 'model') or self.model is None:
            self.load()

        import deepdanbooru.data as ddd

        # convert an image to fit the model
        image_bufs = BytesIO()
        image.save(image_bufs, format='PNG')
        image = ddd.load_image_for_evaluate(
            image_bufs,
            self.model.input_shape[2],
            self.model.input_shape[1]
        )

        image = image.reshape((1, *image.shape[0:3]))

        # evaluate model
        result = self.model.predict(image)

        confidents = result[0].tolist()
        ratings = {}
        tags = {}

        for i, tag in enumerate(self.tags):
            tags[tag] = confidents[i]

        return ratings, tags

class WaifuDiffusionInterrogator(Interrogator):
    def __init__(
        self,
        name: str,
        model_path='model.onnx',
        tags_path='selected_tags.csv',
        **kwargs
    ) -> None:
        super().__init__(name)
        self.model_path = model_path
        self.tags_path = tags_path
        self.kwargs = kwargs

    def download(self) -> Tuple[os.PathLike, os.PathLike]:
        print(f"Loading {self.name} model file from {self.kwargs['repo_id']}")

        model_path = Path(hf_hub_download(
            **self.kwargs, filename=self.model_path))
        tags_path = Path(hf_hub_download(
            **self.kwargs, filename=self.tags_path))
        return model_path, tags_path

    def load(self) -> None:
        model_path, tags_path = self.download()

        # only one of these packages should be installed at a time in any one environment
        # https://onnxruntime.ai/docs/get-started/with-python.html#install-onnx-runtime
        # TODO: remove old package when the environment changes?
        from launch import is_installed, run_pip
        if not is_installed('onnxruntime'):
            package = os.environ.get(
                'ONNXRUNTIME_PACKAGE',
                'onnxruntime-gpu'
            )

            run_pip(f'install {package}', 'onnxruntime')

        from onnxruntime import InferenceSession

        # https://onnxruntime.ai/docs/execution-providers/
        # https://github.com/toriato/stable-diffusion-webui-wd14-tagger/commit/e4ec460122cf674bbf984df30cdb10b4370c1224#r92654958
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if use_cpu:
            providers.pop(0)

        self.model = InferenceSession(str(model_path), providers=providers)

        print(f'Loaded {self.name} model from {model_path}')

        self.tags = pd.read_csv(tags_path)

    def interrogate(
        self,
        image: Image
    ) -> Tuple[
        Dict[str, float],  # rating confidents
        Dict[str, float]  # tag confidents
    ]:
        # init model
        if not hasattr(self, 'model') or self.model is None:
            self.load()

        # code for converting the image and running the model is taken from the link below
        # thanks, SmilingWolf!
        # https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags/blob/main/app.py

        # convert an image to fit the model
        _, height, _, _ = self.model.get_inputs()[0].shape

        # alpha to white
        image = image.convert('RGBA')
        new_image = Image.new('RGBA', image.size, 'WHITE')
        new_image.paste(image, mask=image)
        image = new_image.convert('RGB')
        image = np.asarray(image)

        # PIL RGB to OpenCV BGR
        image = image[:, :, ::-1]

        image = make_square(image, height)
        image = smart_resize(image, height)
        image = image.astype(np.float32)
        image = np.expand_dims(image, 0)

        # evaluate model
        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        confidents = self.model.run([label_name], {input_name: image})[0]

        #S.Gix CUSTOM ( Remove character name )----------
        tags = self.tags[:][['name', 'category']]
        tags['confidents'] = confidents[0]
        
        b_select_only_general_tag = True
        if b_select_only_general_tag:
            tags = tags.loc[tags.category != 4]
        
        tags = tags.drop('category', axis=1)
        #E. Gix CUSTOM-----------------------------------

        #tags = self.tags[:][['name']]
        #tags['confidents'] = confidents[0]

        # first 4 items are for rating (general, sensitive, questionable, explicit)
        ratings = dict(tags[:4].values)

        # rest are regular tags
        tags = dict(tags[4:].values)

        return ratings, tags
#=========================
from modules.shared import models_path
default_ddp_path = Path(models_path, 'deepdanbooru')

#from interrogator import Interrogator, DeepDanbooruInterrogator, WaifuDiffusionInterrogator

interrogators: Dict[str, Interrogator] = {}

def refresh_interrogators() -> List[str]:
    global interrogators
    interrogators = {
        'wd14-convnext-v2': WaifuDiffusionInterrogator( 'wd14-convnext-v2', repo_id='SmilingWolf/wd-v1-4-convnext-tagger-v2', revision='v2.0'),
        'wd14-vit-v2': WaifuDiffusionInterrogator( 'wd14-vit-v2', repo_id='SmilingWolf/wd-v1-4-vit-tagger-v2', revision='v2.0' ),
        'wd14-swinv2-v2': WaifuDiffusionInterrogator( 'wd14-swinv2-v2', repo_id='SmilingWolf/wd-v1-4-swinv2-tagger-v2', revision='v2.0'),
        #'wd14-vit-v2-git': WaifuDiffusionInterrogator( 'wd14-vit-v2-git', repo_id='SmilingWolf/wd-v1-4-vit-tagger-v2'),
        #'wd14-convnext-v2-git': WaifuDiffusionInterrogator( 'wd14-convnext-v2-git', repo_id='SmilingWolf/wd-v1-4-convnext-tagger-v2'),
        #'wd14-swinv2-v2-git': WaifuDiffusionInterrogator( 'wd14-swinv2-v2-git', repo_id='SmilingWolf/wd-v1-4-swinv2-tagger-v2' ),
        'wd14-vit': WaifuDiffusionInterrogator( 'wd14-vit', repo_id='SmilingWolf/wd-v1-4-vit-tagger'),
        'wd14-convnext': WaifuDiffusionInterrogator( 'wd14-convnext', repo_id='SmilingWolf/wd-v1-4-convnext-tagger' ),
    }

    # load deepdanbooru project
    os.makedirs(
        getattr(shared.cmd_opts, 'deepdanbooru_projects_path', default_ddp_path),
        exist_ok=True
    )

    for path in os.scandir(shared.cmd_opts.deepdanbooru_projects_path):
        if not path.is_dir():
            continue

        if not Path(path, 'project.json').is_file():
            continue

        interrogators[path.name] = DeepDanbooruInterrogator(path.name, path)

    return sorted(interrogators.keys())

def split_str(s: str, separator=',') -> List[str]:
    return [x.strip() for x in s.split(separator) if x]
#================================================================================
#E. (External Source) WD Tagger 1.4 =============================================
#================================================================================


#S. Gix Tokenizer ===================================================
class GTokenizer:
    def __init__(self) -> None:
        self.vocabs = None

    def tokenize_word(self, s:str):
        if self.vocabs is None:     
            self.vocabs = []

            path_vocab = os.path.join( BASEDIR, "tokenizer", "vocab_sd.json")                        
            f = open( path_vocab , 'r', encoding="utf8")            
            tokenizer_vocab = json.loads(f.read())
            f.close()
            
            for key in tokenizer_vocab.keys():
                key = key.replace("</w>", "")
                self.vocabs.append(key)

        rlt = 0
        while len(s) > 0:        
            l = len(s)
            for i in range( l , 0 , -1 ):
                s_split = s[0:i]
                if s_split in self.vocabs:
                    rlt += 1
                    s = s[i:]
                    break

        return rlt

    def tokenize(self, str:str):
        rlt = 0
        arr = str.split(" ")
        for s in arr:
            rlt += self.tokenize_word(s)
        return rlt

    def CropPrompt(self, prompt:str, max_token_size:int) -> str:
        a_tags = prompt.strip().split(',')
        for i , s in enumerate(a_tags):            
            a_tags[i] = s.lstrip().rstrip().replace( "_" , " ") #Remove lstrip when need spacing tag

        s_new = []
        tokens = 2 #add base token length (=2) for safety
        for tag in a_tags[:]:
            token_len = self.tokenize(tag) + 1
            if tokens + token_len > max_token_size :
                break
            else:
                tokens += token_len                    
                s_new.append(tag)
        return ', '.join(s_new)

TOKENIZER:GTokenizer = GTokenizer()
#E. Gix Tokenizer===================================================


#S. Detailer Setting Preset===================================================
class GIDPreset():
    base_dir:str #

    def __init__(self, base_dir):
        self.base_dir = str(base_dir)

    def list(self) -> List[str]:
        image_paths = glob.glob(os.path.join(self.base_dir, "*.json"))
        presets = [
            os.path.basename(p) for p in image_paths if os.path.isfile(p)
        ]
        return presets
        
    def save(self, filename:str, configs):
        path = os.path.join( self.base_dir , filename)
        #print( f"Save config path={path}" )
        f = open( path, 'w', encoding="utf8" )
        f.write(json.dumps(configs, indent=4))
        f.close()

    def load(self, filename: str):

        if not filename.endswith('.json'):
            filename += '.json'

        path = os.path.join( self.base_dir, filename)
        #print( f"Load config path={path}" )
        configs = {}
        if os.path.isfile(path):
            f = open( path , 'r', encoding="utf8")            
            configs = json.loads(f.read())
            f.close()

        return configs
#S. Detailer Setting Preset===================================================
PRESETGID = GIDPreset( os.path.join( BASEDIR, "preset" ) )

#S. Gix Detailer ===================================================
class GIDMode(Enum):
    #FULL = 0
    LINEAR = 0
    CHESS = 1
    NONE = 2

class GIDSFMode(Enum):
    NONE = 0
    BAND_PASS = 1
    HALF_TILE = 2
    HALF_TILE_PLUS_INTERSECTIONS = 3

class GIDUpscaler():

    def __init__(self, p, image, upscaler_name, tile_width, tile_height) -> None:
        self.p:StableDiffusionProcessing = p
        self.image:Image = image
        self.scale_factor = math.ceil(max(p.width, p.height) / max(image.width, image.height))
        self.upscaler_name = upscaler_name
        self.redraw = GIDRedraw()        
        self.redraw.tile_width = tile_width if tile_width > 0 else tile_height
        self.redraw.tile_height = tile_height if tile_height > 0 else tile_width        
        self.seams_fix = GIDSeamsFix()        
        self.seams_fix.tile_width = tile_width if tile_width > 0 else tile_height
        self.seams_fix.tile_height = tile_height if tile_height > 0 else tile_width
        
        self.rows = math.ceil(self.p.height / self.redraw.tile_height)
        self.cols = math.ceil(self.p.width / self.redraw.tile_width)
        
    def get_factor(self, num):
        if num == 1:
            return 2
        if num % 4 == 0:
            return 4
        if num % 3 == 0:
            return 3
        if num % 2 == 0:
            return 2
        return 0

    def get_factors(self):
        scales = []
        current_scale = 1
        current_scale_factor = self.get_factor(self.scale_factor)
        while current_scale_factor == 0:
            self.scale_factor += 1
            current_scale_factor = self.get_factor(self.scale_factor)
        while current_scale < self.scale_factor:
            current_scale_factor = self.get_factor(self.scale_factor // current_scale)
            scales.append(current_scale_factor)
            current_scale = current_scale * current_scale_factor
            if current_scale_factor == 0:
                break
        self.scales = enumerate(scales)

    def upscale(self):
        # Log info
        print(f"Canvas size: {self.p.width}x{self.p.height}")
        print(f"Image size: {self.image.width}x{self.image.height}")
        print(f"Scale factor: {self.scale_factor}")
        # Check upscaler is not empty
        if self.upscaler_name == "None":
            self.image = self.image.resize((self.p.width, self.p.height), resample=Image.LANCZOS)
            return
        
        upscaler = list(filter(lambda u: u.name == self.upscaler_name, shared.sd_upscalers))[0]        
        # Get list with scale factors
        self.get_factors()                
        # Upscaling image over all factors
        for index, value in self.scales:            
            #print(f"Upscaling iteration {index+1} with scale factor {value}")                        
            self.image = upscaler.scaler.upscale(self.image, value, upscaler.data_path)
        # Resize image to set values
        self.image = self.image.resize((self.p.width, self.p.height), resample=Image.LANCZOS)

    def setup_redraw(self, redraw_mode_enum, redraw_full_res, padding, mask_blur, redraw_denoise, redraw_random_seed, redraw_steps, redraw_cfg, show_redraw_steps, 
                     use_partial_prompt, tagger):
        
        self.redraw.mode = redraw_mode_enum
        self.redraw.redraw_full_res = redraw_full_res
        self.redraw.enabled = self.redraw.mode != GIDMode.NONE
        self.redraw.padding = padding
        self.redraw.redraw_denoise = redraw_denoise
        self.redraw.redraw_random_seed = redraw_random_seed
        self.redraw.redraw_steps = redraw_steps
        self.redraw.redraw_cfg = redraw_cfg

        self.p.mask_blur = mask_blur

        self.redraw.show_redraw_steps = show_redraw_steps

        self.redraw.use_partial_prompt = use_partial_prompt
        if use_partial_prompt:
            self.redraw.tagger = tagger            
            
        

    def setup_seams_fix(self, padding, denoise, mask_blur, width, mode):
        self.seams_fix.padding = padding
        self.seams_fix.denoise = denoise
        self.seams_fix.mask_blur = mask_blur
        self.seams_fix.width = width
        self.seams_fix.mode = GIDSFMode(mode)
        self.seams_fix.enabled = self.seams_fix.mode != GIDSFMode.NONE

    def save_image(self, _info, img = None, suffix="" ):
        if img is None :
            img = self.image
        images.save_image(img, self.p.outpath_samples, "", self.p.seed, self.p.prompt, opts.grid_format, info=_info, p=self.p, suffix=suffix)

    def calc_jobs_count(self):
        redraw_job_count = (self.rows * self.cols) if self.redraw.enabled else 0
        seams_job_count = 0
        if self.seams_fix.mode == GIDSFMode.BAND_PASS:
            seams_job_count = self.rows + self.cols - 2
        elif self.seams_fix.mode == GIDSFMode.HALF_TILE:
            seams_job_count = self.rows * (self.cols - 1) + (self.rows - 1) * self.cols
        elif self.seams_fix.mode == GIDSFMode.HALF_TILE_PLUS_INTERSECTIONS:
            seams_job_count = self.rows * (self.cols - 1) + (self.rows - 1) * self.cols + (self.rows - 1) * (self.cols - 1)

        return redraw_job_count + seams_job_count        

    def print_info(self):
        print(f"Tiles size={self.redraw.tile_width}x{self.redraw.tile_height}")
        print(f"Tiles Grid: {self.rows}x{self.cols} = {self.rows * self.cols}")
        print(f"Redraw enabled: {self.redraw.enabled}, denoise: {self.redraw.redraw_denoise}")
        print(f"Seams fix mode: {self.seams_fix.mode.name}")

class GIDRedraw():
    def __init__(self): 
        self.redraw_full_res = True       
        self.show_redraw_steps = False
        self.use_partial_prompt = False
        self.tagger:GIDTagger = None
        self.use_only_inputed_prompt = False        
        self.logs = []
        self.image_width = 0
        self.image_height = 0

    def init_draw(self, p, width, height):  
        self.image_width = width
        self.image_height = height
        p.inpaint_full_res = self.redraw_full_res
        p.inpaint_full_res_padding = self.padding
        mask = Image.new("L", (width, height), "black")
        draw = ImageDraw.Draw(mask)        
        return mask, draw

    def calc_rectangle(self, xi, yi):
        x1 = xi * self.tile_width
        y1 = yi * self.tile_height
        x2 = min(self.image_width, xi * self.tile_width + self.tile_width)
        y2 = min(self.image_height , yi * self.tile_height + self.tile_height)

        return x1, y1, x2, y2


    def start(self, p, image, rows, cols, result_images):
        self.initial_info = None

        #S. backup and set. ========================
        bak_denoising_strength = p.denoising_strength
        bak_prompt = p.prompt
        bak_seed = p.seed
        bak_subseed = p.subseed
        bak_steps = p.steps
        bak_cfg_scale = p.cfg_scale
        #E. backup and set. ========================
        rlt = None
        if self.mode == GIDMode.LINEAR:
            rlt = self.linear_process(p, image, bak_prompt, rows, cols, result_images)
        elif self.mode == GIDMode.CHESS:
            rlt = self.chess_process(p, image, bak_prompt, rows, cols, result_images)
        #S. restore====================
        p.denoising_strength = bak_denoising_strength
        p.prompt = bak_prompt
        if self.redraw_random_seed:
            p.seed = bak_seed
            p.subseed = bak_subseed
        p.steps = bak_steps
        p.cfg_scale = bak_cfg_scale
        #E. restore====================

        return rlt
    

    def draw_partial(self, image, draw, mask, xi , yi , p , prompt_original, result_images:List ):        
        rect = self.calc_rectangle(xi, yi)
        draw.rectangle(rect, fill="white")

        image_sliced = image.crop(rect)
        if self.use_partial_prompt == True:
            p.prompt = self.tagger.GetPartialPrompts( image_sliced, prompt_original )            

        if p.inpaint_full_res: 
            p.width = math.ceil((self.tile_width+self.padding) / 64) * 64
            p.height = math.ceil((self.tile_height+self.padding) / 64) * 64     
        else:
            p.width = self.image_width
            p.height = self.image_height

        b_draw_and_paste = False
        if b_draw_and_paste:
            p.init_images = [ image_sliced ]
            p.width = image_sliced.width
            p.height = image_sliced.height        
        else:
            p.init_images = [image]
            p.image_mask = mask
            
        #if p.inpaint_full_res:            
            #p.width = math.ceil((self.tile_width+self.padding) / 64) * 64
            #p.height = math.ceil((self.tile_height+self.padding) / 64) * 64            
        #else:            
            #p.width = width
            #p.height = height
            #p.width = math.ceil((self.tile_width+self.padding) / 64) * 64
            #p.height = math.ceil((self.tile_height+self.padding) / 64) * 64            

        if self.redraw_random_seed:
            p.seed = -1
            p.subseed = -1
            processing.fix_seed(p)     
        
        s_log = f"Partial Drawing. (col={xi}, row={yi}), Rect (x{rect[0]} y{rect[1]}) to (x{rect[2]}, y{rect[3]}),\n" + \
            f"seed : {p.seed} , subseed : {p.subseed}, steps : {p.steps}, cfg : {p.cfg_scale}  \n" + \
            f"Prompt={p.prompt}"
        
        print(s_log)
        self.logs.append( s_log )

        processed = processing.process_images(p)
        draw.rectangle(rect, fill="black")     

        if (len(processed.images) > 0):
            image_redraw = processed.images[0]
            
            if b_draw_and_paste:
                overlay = Image.new('RGBA', (image.width, image.height))
                overlay.paste(image_redraw, (rect[0], rect[1]))

                image = image.convert('RGBA')
                image.alpha_composite(overlay)
                image = image.convert('RGB')
            else:
                image = image_redraw

            if self.show_redraw_steps:
                for img in reversed(processed.images):
                    result_images.insert(0, img)
        return image
        

    def linear_process(self, p, image, prompt_original, rows, cols, result_images):
        
        p.denoising_strength = self.redraw_denoise
        if self.redraw_steps > 0:
            p.steps = self.redraw_steps
        if self.redraw_cfg > 0:
            p.cfg_scale = self.redraw_cfg
        
        mask, draw = self.init_draw(p, image.width, image.height)
        for yi in range(rows):
            for xi in range(cols):
                if state.interrupted:
                    break                               
                image = self.draw_partial( image, draw, mask, xi, yi, p , prompt_original, result_images )
        
        p.width = image.width
        p.height = image.height
        #self.initial_info = processed.infotext(p, 0)


        return image

    def chess_process(self, p, image, prompt_original, rows, cols, result_images):        
        p.denoising_strength = self.redraw_denoise
        if self.redraw_steps > 0:
            p.steps = self.redraw_steps
        if self.redraw_cfg > 0:
            p.cfg_scale = self.redraw_cfg

        mask, draw = self.init_draw(p, image.width, image.height)
        tiles = []
        # calc tiles colors
        for yi in range(rows):
            for xi in range(cols):
                if state.interrupted:
                    break
                if xi == 0:
                    tiles.append([])
                color = xi % 2 == 0
                if yi > 0 and yi % 2 != 0:
                    color = not color
                tiles[yi].append(color)

        for yi in range(len(tiles)):
            for xi in range(len(tiles[yi])):
                if state.interrupted:
                    break
                if not tiles[yi][xi]:
                    tiles[yi][xi] = not tiles[yi][xi]
                    continue
                tiles[yi][xi] = not tiles[yi][xi]
                
                image = self.draw_partial( image, draw, mask, xi, yi, p , prompt_original, result_images )

        for yi in range(len(tiles)):
            for xi in range(len(tiles[yi])):
                if state.interrupted:
                    break
                if not tiles[yi][xi]:
                    continue                
                image = self.draw_partial( image, draw, mask, xi, yi, p , prompt_original, result_images )
                
        p.width = image.width
        p.height = image.height
        #self.initial_info = processed.infotext(p, 0)
        return image

class GIDSeamsFix():
    def __init__(self):
        pass

    def init_draw(self, p):
        self.initial_info = None
        p.width = math.ceil((self.tile_width+self.padding) / 64) * 64
        p.height = math.ceil((self.tile_height+self.padding) / 64) * 64

    def half_tile_process(self, p, image, rows, cols):

        self.init_draw(p)
        processed = None

        gradient = Image.linear_gradient("L")
        row_gradient = Image.new("L", (self.tile_width, self.tile_height), "black")
        row_gradient.paste(gradient.resize(
            (self.tile_width, self.tile_height//2), resample=Image.BICUBIC), (0, 0))
        row_gradient.paste(gradient.rotate(180).resize(
                (self.tile_width, self.tile_height//2), resample=Image.BICUBIC),
                (0, self.tile_height//2))
        col_gradient = Image.new("L", (self.tile_width, self.tile_height), "black")
        col_gradient.paste(gradient.rotate(90).resize(
            (self.tile_width//2, self.tile_height), resample=Image.BICUBIC), (0, 0))
        col_gradient.paste(gradient.rotate(270).resize(
            (self.tile_width//2, self.tile_height), resample=Image.BICUBIC), (self.tile_width//2, 0))

        p.denoising_strength = self.denoise
        p.mask_blur = self.mask_blur

        for yi in range(rows-1):
            for xi in range(cols):
                if state.interrupted:
                    break
                p.width = self.tile_width
                p.height = self.tile_height
                p.inpaint_full_res = True
                p.inpaint_full_res_padding = self.padding
                mask = Image.new("L", (image.width, image.height), "black")
                mask.paste(row_gradient, (xi*self.tile_width, yi*self.tile_height + self.tile_height//2))

                p.init_images = [image]
                p.image_mask = mask
                processed = processing.process_images(p)
                if (len(processed.images) > 0):
                    image = processed.images[0]

        for yi in range(rows):
            for xi in range(cols-1):
                if state.interrupted:
                    break
                p.width = self.tile_width
                p.height = self.tile_height
                p.inpaint_full_res = True
                p.inpaint_full_res_padding = self.padding
                mask = Image.new("L", (image.width, image.height), "black")
                mask.paste(col_gradient, (xi*self.tile_width+self.tile_width//2, yi*self.tile_height))

                p.init_images = [image]
                p.image_mask = mask
                processed = processing.process_images(p)
                if (len(processed.images) > 0):
                    image = processed.images[0]

        p.width = image.width
        p.height = image.height
        if processed is not None:
            self.initial_info = processed.infotext(p, 0)

        return image

    def half_tile_process_corners(self, p, image, rows, cols):
        fixed_image = self.half_tile_process(p, image, rows, cols)
        processed = None
        self.init_draw(p)
        gradient = Image.radial_gradient("L").resize(
            (self.tile_width, self.tile_height), resample=Image.BICUBIC)
        gradient = ImageOps.invert(gradient)
        p.denoising_strength = self.denoise
        #p.mask_blur = 0
        p.mask_blur = self.mask_blur

        for yi in range(rows-1):
            for xi in range(cols-1):
                if state.interrupted:
                    break
                p.width = self.tile_width
                p.height = self.tile_height
                p.inpaint_full_res = True
                p.inpaint_full_res_padding = 0
                mask = Image.new("L", (fixed_image.width, fixed_image.height), "black")
                mask.paste(gradient, (xi*self.tile_width + self.tile_width//2,
                                      yi*self.tile_height + self.tile_height//2))

                p.init_images = [fixed_image]
                p.image_mask = mask
                processed = processing.process_images(p)
                if (len(processed.images) > 0):
                    fixed_image = processed.images[0]

        p.width = fixed_image.width
        p.height = fixed_image.height
        if processed is not None:
            self.initial_info = processed.infotext(p, 0)

        return fixed_image

    def band_pass_process(self, p, image, cols, rows):

        self.init_draw(p)
        processed = None

        p.denoising_strength = self.denoise
        p.mask_blur = 0

        gradient = Image.linear_gradient("L")
        mirror_gradient = Image.new("L", (256, 256), "black")
        mirror_gradient.paste(gradient.resize((256, 128), resample=Image.BICUBIC), (0, 0))
        mirror_gradient.paste(gradient.rotate(180).resize((256, 128), resample=Image.BICUBIC), (0, 128))

        row_gradient = mirror_gradient.resize((image.width, self.width), resample=Image.BICUBIC)
        col_gradient = mirror_gradient.rotate(90).resize((self.width, image.height), resample=Image.BICUBIC)

        for xi in range(1, rows):
            if state.interrupted:
                    break
            p.width = self.width + self.padding * 2
            p.height = image.height
            p.inpaint_full_res = True
            p.inpaint_full_res_padding = self.padding
            mask = Image.new("L", (image.width, image.height), "black")
            mask.paste(col_gradient, (xi * self.tile_width - self.width // 2, 0))

            p.init_images = [image]
            p.image_mask = mask
            processed = processing.process_images(p)
            if (len(processed.images) > 0):
                image = processed.images[0]
        for yi in range(1, cols):
            if state.interrupted:
                    break
            p.width = image.width
            p.height = self.width + self.padding * 2
            p.inpaint_full_res = True
            p.inpaint_full_res_padding = self.padding
            mask = Image.new("L", (image.width, image.height), "black")
            mask.paste(row_gradient, (0, yi * self.tile_height - self.width // 2))

            p.init_images = [image]
            p.image_mask = mask
            processed = processing.process_images(p)
            if (len(processed.images) > 0):
                image = processed.images[0]

        p.width = image.width
        p.height = image.height
        if processed is not None:
            self.initial_info = processed.infotext(p, 0)

        return image

    def start(self, p, image, rows, cols):
        if GIDSFMode(self.mode) == GIDSFMode.BAND_PASS:
            return self.band_pass_process(p, image, rows, cols)
        elif GIDSFMode(self.mode) == GIDSFMode.HALF_TILE:
            return self.half_tile_process(p, image, rows, cols)
        elif GIDSFMode(self.mode) == GIDSFMode.HALF_TILE_PLUS_INTERSECTIONS:
            return self.half_tile_process_corners(p, image, rows, cols)
        else:
            return image


class GIDTagger():
    def __init__(self): 
        self.use_only_inputed_prompt = False
        self.max_token = 0
        self.add_tags = ""
        self.exclude_tags = ""
        
    def setup(self, use_only_inputed_prompt, interrogator_name, interrogator_threshold, interrogator_add_tags, interrogator_exclude_tags, interrogator_max_token):
        self.use_only_inputed_prompt = use_only_inputed_prompt
        self.max_token = interrogator_max_token
        self.add_tags = interrogator_add_tags
        self.exclude_tags = interrogator_exclude_tags

        self.interrogator: Interrogator = interrogators[interrogator_name]
        self.postprocess_opts = ( interrogator_threshold, split_str(self.add_tags), [], False, False, True, [], False )
        
    def on_interrogate(self, image):        
        ratings, tags = self.interrogator.interrogate(image)
        processed_tags = Interrogator.postprocess_tags(
            tags,
            *self.postprocess_opts
        )
        return processed_tags
    
    def GetPartialPrompts(self, image_sliced, prompt_original):
        exclude = split_str(self.exclude_tags) if self.exclude_tags is not None else []
        dic_tags = self.on_interrogate(image_sliced)
        arr_tag = []
        for k in dic_tags.keys():  
            b_exclude = False
            for s_ex in exclude:
                s_ex = s_ex.strip()                    
                if len(s_ex) < 1:
                    continue
                if "*" in s_ex:
                    s_ex = s_ex.replace(';', '\;').replace(')', '\)').replace('(', '\(').replace( "*" , "[a-z]+")
                    if len(s_ex) > 0:
                        re_pattern = re.compile(s_ex , re.IGNORECASE)
                        if re_pattern.match(k) is not None :
                            print(f"Exclude tag '{k}'")
                            b_exclude = True
                            break    
                elif s_ex == k:
                    b_exclude = True
                    break

            if b_exclude == True:
                continue    

            if self.use_only_inputed_prompt:
                add_tags = split_str(self.add_tags)
                if k in prompt_original or k in add_tags:
                    arr_tag.append(k)
            else:
                arr_tag.append(k)

        prompt_temp = ", ".join( arr_tag )
        if self.max_token > 0:                
            #print( f"Original prompt={prompt_temp}")
            prompt_temp = TOKENIZER.CropPrompt( prompt_temp, self.max_token )
            #print( f"Cropped prompt={prompt_temp}")
        return prompt_temp

class Script(scripts.Script):
    def title(self):
        return "Gix Detailer"

    def show(self, is_img2img):
        return True
    
    #gradio function
    def ConfigSave(self, filename:str, *values):
        if filename is None:
            return []
        if filename.endswith(".json") == False:
            filename += ".json"
        configs = {}
        idx = 0        
        for k , cp in self.components_dict.items(): 
            v = values[idx]
            vsb = True
            if hasattr(cp, "visible"):
                vsb = getattr(cp, "visible")
            configs[k] = {"value":v , "visible":vsb }
            idx += 1            
        PRESETGID.save( filename, configs )
        return []

    #gradio function
    def ConfigLoad(self, filename):        
        cfg = PRESETGID.load( filename )

        rlt = []
        for cp in self.components_list:            
            if cfg is None:
                rlt.append( gr.update() )            
                continue
            
            b_added = False
            for k , cp_in in self.components_dict.items(): 
                if cp != cp_in :
                    continue
                if k in cfg.keys():
                    cfg_row = cfg[k]                    
                    v = cfg_row["value"] if "value" in cfg_row.keys() else None                    
                    vsb = cfg_row["visible"] if "visible" in cfg_row.keys() else True
                    
                    if hasattr(cp, 'type') and cp.type == "index":
                        idx = 0
                        try:
                            idx = int(v)
                            if hasattr(cp, "choices"):
                                if len(cp.choices) < 1:
                                    v = None        
                                else:
                                    if idx >= len(cp.choices):
                                        idx = 0
                                    v = cp.choices[idx]
                        except:
                            v = None                          
                    rlt.append( gr.update(value=v, visible=vsb) )                        
                    b_added = True
                break
            if b_added == False:
                rlt.append( gr.update() )
        #print( len(rlt) , len(self.components_list))
        return rlt
                
    def ui(self, is_img2img):
        
        cfg = PRESETGID.load( "default.json" )

        target_size_types = [
            "From img2img2 settings",
            "Custom size",
            "Scale from image size"
        ]

        redraw_modes = [ "Linear", "Chess", "None" ]

        seams_fix_types = [
            "None",
            "Band pass",
            "Half tile offset pass",
            "Half tile offset pass + intersections"
        ]        
        #gr.HTML("<p style=\"margin-bottom:0.75em\"></p>")
        def SetV(cfg, name , v ):
            return cfg[name]["value"] if name in cfg and "value" in cfg[name] else v
        
        dummy_component = gr.Label(visible=False)

        with gr.Accordion("Gix Detailer", open=False, elem_id="gixdetailer"):
            
            #=============================================================
            gr.HTML(value="<p>" + "Click 'Load' button" + "</p>")
            
            with FormRow(variant="compact"):
                available_presets = PRESETGID.list()
                selected_preset = gr.Dropdown(
                    label='Preset',
                    choices=available_presets,
                    value=available_presets[0],
                    interactive=True
                )
                ui.create_refresh_button( selected_preset, lambda: None, lambda: {'choices': PRESETGID.list()}, 'refresh_preset')
                
                button_load_preset = gr.Button( "Load" )
                button_save_preset = gr.Button( "Save" )                    
            #print(f"Load config. {cfg}") #load when first loading WEBUI
            #=============================================================

            gr.HTML(value="<p style='font-size:1.2em;font-weight: bold;'>" + "General setting" + "</p>")            
            
            with FormRow(variant="compact"):    
                disable_controlnet_during_detailup = gr.Checkbox(label="Disable Control-Net during details-up (on Hires.fix, R-Detailer, Seams fix, D-Detailer) ※ Recommended", value=SetV(cfg, "disable_controlnet_during_detailup", True))
            with FormRow(variant="compact"):    
                show_redraw_steps = gr.Checkbox(label="(UI Only) Show redraw steps.", value=SetV(cfg, "show_redraw_steps", False))
                
            gr.HTML(value="<p style='font-size:1.2em;font-weight: bold;'>" + "Upscales" + "</p>")            
            with FormRow(variant="compact"):
                target_size_type = gr.Dropdown(label="Upscale by", choices=[k for k in target_size_types], type="index",
                                    value=target_size_types[ SetV(cfg, "target_size_type", 2)] )

                custom_width = gr.Slider(label='Custom width', minimum=64, maximum=8192, step=64, value=SetV(cfg , "custom_width", 2048), visible=False, interactive=True)
                custom_height = gr.Slider(label='Custom height', minimum=64, maximum=8192, step=64, value=SetV(cfg , "custom_height", 2048), visible=False, interactive=True)
                custom_scale = gr.Slider(label='Scales to', minimum=1, maximum=16, step=0.01, value=SetV(cfg , "custom_scale", 2), visible=True, interactive=True)
            with FormRow(variant="compact"):    
                upscaler_name = gr.Dropdown(label='Select upscaler', choices=[*[x.name for x in shared.sd_upscalers]], value=shared.latent_upscale_default_mode)                
            with FormRow(variant="compact"):
                save_image_upscale = gr.Checkbox(label="Save image in 'Upscale' stage", value=SetV(cfg, "save_image_upscale", False))
            #=============================================================
            gr.HTML(value="<p style='font-size:1.2em;font-weight: bold;'>" + "Hires. Fix" + "</p>")
            with FormRow(variant="compact"):
                save_image_hires_fix = gr.Checkbox(label="Save image in 'Hires. fix' stage", value=SetV(cfg, "save_image_hires_fix", True))
            with FormRow(variant="compact"):
                hiresfix_denoise = gr.Slider(label='Denoising strength', minimum=0, maximum=1, step=0.1, value=SetV(cfg, "hiresfix_denoise", 0.4))
                hiresfix_random_seed = gr.Checkbox(label="Random seed", value=SetV(cfg, "hiresfix_random_seed", True))
            with FormRow(variant="compact"):
                hiresfix_steps = gr.Slider(label='Sampling steps (0 = override)', minimum=0, maximum=150, step=1, value=SetV(cfg, "hiresfix_steps", 0))
                hiresfix_cfg = gr.Slider(label='CFG Scale (0 = override)', minimum=0, maximum=30, step=0.25, value=SetV(cfg, "hiresfix_cfg", 0))
            #=============================================================
            gr.HTML(value="<p style='font-size:1.2em;font-weight: bold;'>" + "Redraw Detailer" + "</p>")                                
            with FormRow(variant="compact"):                    
                redraw_mode = gr.Dropdown(label="Redraw method.", choices=[k for k in redraw_modes], type="index", value=redraw_modes[SetV(cfg, "redraw_mode",0)])
            with FormRow(variant="compact"):
                save_image_detailed = gr.Checkbox(label="Save image in 'Detailer' stage", value=SetV(cfg, "save_image_detailed",True))
            with FormGroup(variant="compact"):
                redraw_on_hires_fix = gr.Checkbox(label="Detail up based on 'Hires. fix' image.", value=SetV(cfg, "redraw_on_hires_fix", True))
                redraw_full_res = gr.Checkbox(label="Redraw inpaint at full resolution", value=SetV(cfg, "redraw_full_res", True))
            with FormRow(variant="compact"):
                redraw_denoise = gr.Slider(label='Denoising strength', minimum=0, maximum=1, step=0.1, value=SetV(cfg, "redraw_denoise", 0.4))
                redraw_random_seed = gr.Checkbox(label="Random seed", value=SetV(cfg, "redraw_random_seed", True))                
            with FormRow(variant="compact"):
                redraw_steps = gr.Slider(label='Sampling steps (0 = override)', minimum=0, maximum=150, step=1, value=SetV(cfg, "redraw_steps", 0))
                redraw_cfg = gr.Slider(label='CFG Scale (0 = override)', minimum=0, maximum=30, step=0.25, value=SetV(cfg, "redraw_cfg", 0))
            with FormRow(variant="compact"):    
                tile_width = gr.Slider(minimum=0, maximum=2048, step=64, label='Tile width.', value=SetV(cfg, "tile_width", 768))
                tile_height = gr.Slider(minimum=0, maximum=2048, step=64, label='Tile height.', value=SetV(cfg, "tile_height", 0))
            with FormRow(variant="compact"):
                mask_blur = gr.Slider(label='Mask blur.', minimum=0, maximum=64, step=1, value=SetV(cfg, "mask_blur", 8))
                padding = gr.Slider(label='Padding.', minimum=0, maximum=128, step=1, value=SetV(cfg, "padding", 32))           
            

            #=============================================================
            gr.HTML(value="<p style='font-size:1.2em;font-weight: bold;'>" + "Redraw Detailer with partial Prompt" + "</p>")
            with FormRow(variant="compact"):
                use_partial_prompt = gr.Checkbox(label="Use 'partial prompt' to redraw each tile.", value=SetV(cfg, "use_partial_prompt", True))

            with FormRow(variant="compact"):                             
                interrogator_names = refresh_interrogators()
                interrogator_name = gr.Dropdown( label='Interrogator', choices=interrogator_names,
                    value=( None if len(interrogator_names) < 1 else SetV(cfg, "interrogator_name", "") )
                )
                interrogator_threshold = gr.Slider( label='Tagger Threshold', minimum=0, maximum=1, value=SetV(cfg, "interrogator_threshold", 0.35))
            with FormRow(variant="compact"):
                use_only_inputed_prompt = gr.Checkbox(label="Prompt only you inputed", value=SetV(cfg, "use_only_inputed_prompt", False))
                interrogator_max_token = gr.Dropdown(label="Token limit", choices=["75" , "150" , "225", "Unlimited"], 
                                                        type="value", value=SetV(cfg, "interrogator_max_token", "75"), 
                                                        elem_id="gix_detailer_gr_" + "interrogator_max_token" )
            with FormRow(variant="compact"):
                interrogator_add_tags = gr.Textbox(label="Additional tags (split by comma)", 
                                                value=SetV(cfg, "interrogator_add_tags", ""), 
                                                placeholder="",
                                                elem_id="gix_detailer_gr_" + "interrogator_add_tags" )
                
            with FormRow(variant="compact"):
                interrogator_exclude_tags = gr.Textbox(label='Exclude tags (split by comma) (* = any word)', value=SetV(cfg, "interrogator_exclude_tags", ""),
                                                    placeholder="ex) 2girls, multiple girls, white background, simple background, black background", elem_id="gix_detailer_gr_" + "interrogator_exclude_tags")
                
            #=============================================================
            gr.HTML(value="<p style='font-size:1.2em;font-weight: bold;'>" + "Seams fix" + "</p>")                
            with FormRow(variant="compact"):
                seams_fix_type = gr.Dropdown(label="Type", choices=[k for k in seams_fix_types], type="index", value=seams_fix_types[SetV(cfg, "seams_fix_type", 0)] )
            with FormRow(variant="compact"):
                save_image_seams_fix = gr.Checkbox(label="Save image in 'Seams fix' stage", value=SetV(cfg, "save_image_seams_fix", False))
            with FormRow(variant="compact"):
                seams_fix_denoise = gr.Slider(label='denoising strength', minimum=0, maximum=1, step=0.01, value=SetV(cfg, "seams_fix_denoise", 0.35), visible=False, interactive=True)
                seams_fix_width = gr.Slider(label='Width', minimum=0, maximum=128, step=1, value=SetV(cfg, "seams_fix_width",64), visible=False, interactive=True)
                seams_fix_mask_blur = gr.Slider(label='Mask blur', minimum=0, maximum=64, step=1, value=SetV(cfg, "seams_fix_mask_blur",4), visible=False, interactive=True)
                seams_fix_padding = gr.Slider(label='Padding', minimum=0, maximum=128, step=1, value=SetV(cfg, "seams_fix_padding", 16), visible=False, interactive=True)                
            #=============================================================

            #=============================================================
            gr.HTML(value="<p style='font-size:1.2em;font-weight: bold;'>" + "Detection Detailer" + "</p>")  

            import modules.ui

            model_list = dd_list_models(dd_models_path)
            model_list.insert(0, "None")
            
            with FormRow(variant="compact"):
                save_image_ddetailer = gr.Checkbox(label="Save image in 'Detection Detailer' stage", value=SetV(cfg, "save_image_ddetailer", True), visible=False)
            
            with FormRow(variant="compact"):                
                dd_denoising_strength = gr.Slider(label='Denoising strength', minimum=0.0, maximum=1.0, step=0.01, value=0.4, visible=False)
                dd_random_seed = gr.Checkbox(label="Random seed", value=SetV(cfg, "dd_random_seed", True))                
            
            with FormRow(variant="compact"):
                dd_mask_blur = gr.Slider(label='Mask blur', minimum=0, maximum=64, step=1, value=4, visible=False)                
                dd_inpaint_full_res_padding = gr.Slider(label='Padding', minimum=0, maximum=256, step=4, value=32, visible=False)
            with FormRow(variant="compact"):
                dd_inpaint_full_res = gr.Checkbox(label='Inpaint at full resolution', value=True, visible=False)

            with FormRow(variant="compact"):
                dd_model_a = gr.Dropdown(label="Primary detection model (A)", choices=model_list,value = "None", visible=True, type="value")
            
            with FormRow(variant="compact"):
                dd_conf_a = gr.Slider(label='Detection confidence threshold % (A)', minimum=0, maximum=100, step=1, value=30, visible=False)
                dd_dilation_factor_a = gr.Slider(label='Dilation factor (A)', minimum=0, maximum=255, step=1, value=4, visible=False)

            with FormRow(variant="compact"):
                dd_offset_x_a = gr.Slider(label='X offset (A)', minimum=-200, maximum=200, step=1, value=0, visible=False)
                dd_offset_y_a = gr.Slider(label='Y offset (A)', minimum=-200, maximum=200, step=1, value=0, visible=False)
            
            with FormRow(variant="compact"):
                dd_preprocess_b = gr.Checkbox(label='Inpaint model B detections before model A runs', value=False, visible=False)
                dd_bitwise_op = gr.Radio(label='Bitwise operation', choices=['None', 'A&B', 'A-B'], value="None", visible=False)  
           
            with FormRow(variant="compact"):
                dd_model_b = gr.Dropdown(label="Secondary detection model (B) (optional)", choices=model_list,value = "None", visible =False, type="value")

            with FormRow(variant="compact"):
                dd_conf_b = gr.Slider(label='Detection confidence threshold % (B)', minimum=0, maximum=100, step=1, value=30, visible=False)
                dd_dilation_factor_b = gr.Slider(label='Dilation factor (B)', minimum=0, maximum=255, step=1, value=4, visible=False)
            
            with FormRow(variant="compact"):
                dd_offset_x_b = gr.Slider(label='X offset (B)', minimum=-200, maximum=200, step=1, value=0, visible=False)
                dd_offset_y_b = gr.Slider(label='Y offset (B)', minimum=-200, maximum=200, step=1, value=0, visible=False)
        
            

            dd_model_a.change(
                lambda modelname: {
                    save_image_ddetailer:dd_gr_show( modelname != "None" ),
                    dd_model_b:dd_gr_show( modelname != "None" ),
                    dd_conf_a:dd_gr_show( modelname != "None" ),
                    dd_dilation_factor_a:dd_gr_show( modelname != "None"),
                    dd_offset_x_a:dd_gr_show( modelname != "None" ),
                    dd_offset_y_a:dd_gr_show( modelname != "None" ),
                    dd_mask_blur:dd_gr_show( modelname != "None" ),
                    dd_denoising_strength:dd_gr_show( modelname != "None" ),
                    dd_random_seed:dd_gr_show( modelname != "None" ),                    
                    dd_inpaint_full_res:dd_gr_show( modelname != "None" ),
                    dd_inpaint_full_res_padding:dd_gr_show( modelname != "None" ),
                },
                inputs=[dd_model_a],
                outputs=[save_image_ddetailer, dd_model_b, dd_conf_a, dd_dilation_factor_a, dd_offset_x_a, dd_offset_y_a, 
                        dd_mask_blur, dd_denoising_strength, dd_random_seed, dd_inpaint_full_res, dd_inpaint_full_res_padding]
            )

            dd_model_b.change(
                lambda modelname: {
                    dd_preprocess_b:dd_gr_show( modelname != "None" ),
                    dd_bitwise_op:dd_gr_show( modelname != "None" ),
                    dd_conf_b:dd_gr_show( modelname != "None" ),
                    dd_dilation_factor_b:dd_gr_show( modelname != "None"),
                    dd_offset_x_b:dd_gr_show( modelname != "None" ),
                    dd_offset_y_b:dd_gr_show( modelname != "None" )
                },
                inputs=[dd_model_b],
                outputs=[dd_preprocess_b, dd_bitwise_op, dd_conf_b, dd_dilation_factor_b, dd_offset_x_b, dd_offset_y_b]
            )

        self.components_dict = {}
        self.components_list = []

        self.components_dict["disable_controlnet_during_detailup"] = disable_controlnet_during_detailup          
        self.components_dict["show_redraw_steps"] = show_redraw_steps          

        self.components_dict["target_size_type"] = target_size_type               
        self.components_dict["custom_width"] =  custom_width                   
        self.components_dict["custom_height"] = custom_height                 
        self.components_dict["custom_scale"] = custom_scale                  
        self.components_dict["upscaler_name"] =  upscaler_name   
        self.components_dict["save_image_upscale"] =  save_image_upscale
        
        self.components_dict["save_image_hires_fix"] = save_image_hires_fix   
        self.components_dict["hiresfix_denoise"] = hiresfix_denoise  
        self.components_dict["hiresfix_random_seed"] = hiresfix_random_seed   
        self.components_dict["hiresfix_steps"] = hiresfix_steps   
        self.components_dict["hiresfix_cfg"] = hiresfix_cfg

        self.components_dict["redraw_mode"] = redraw_mode      
        self.components_dict["redraw_denoise"] = redraw_denoise                
        self.components_dict["redraw_random_seed"] = redraw_random_seed
        self.components_dict["redraw_steps"] = redraw_steps
        self.components_dict["redraw_cfg"] = redraw_cfg
        
        self.components_dict["redraw_on_hires_fix"] = redraw_on_hires_fix 
        self.components_dict["redraw_full_res"] = redraw_full_res
        self.components_dict["tile_width"] =  tile_width                          
        self.components_dict["tile_height"] = tile_height                        
        self.components_dict["mask_blur"] = mask_blur                        
        self.components_dict["padding"] = padding                           
        
        self.components_dict["use_partial_prompt"] = use_partial_prompt           
        self.components_dict["use_only_inputed_prompt"] =  use_only_inputed_prompt  
        self.components_dict["interrogator_name"] = interrogator_name            
        self.components_dict["interrogator_threshold"] =  interrogator_threshold    
        self.components_dict["interrogator_max_token"] =  interrogator_max_token
        self.components_dict["interrogator_add_tags"] =  interrogator_add_tags         
        self.components_dict["interrogator_exclude_tags"] =  interrogator_exclude_tags         
        self.components_dict["seams_fix_type"] = seams_fix_type                
        self.components_dict["seams_fix_denoise"] =  seams_fix_denoise           
        self.components_dict["seams_fix_width"] = seams_fix_width               
        self.components_dict["seams_fix_mask_blur"] =  seams_fix_mask_blur        
        self.components_dict["seams_fix_padding"] =  seams_fix_padding           
        self.components_dict["save_image_detailed"] =  save_image_detailed       
        self.components_dict["save_image_seams_fix"] = save_image_seams_fix
        
        self.components_dict["save_image_ddetailer"] = save_image_ddetailer 
        self.components_dict["dd_model_a"] = dd_model_a 
        self.components_dict["dd_conf_a"] = dd_conf_a 
        self.components_dict["dd_dilation_factor_a"] = dd_dilation_factor_a 
        self.components_dict["dd_offset_x_a"] = dd_offset_x_a 
        self.components_dict["dd_offset_y_a"] = dd_offset_y_a 
        self.components_dict["dd_preprocess_b"] = dd_preprocess_b 
        self.components_dict["dd_bitwise_op"] = dd_bitwise_op 
        self.components_dict["dd_model_b"] = dd_model_b 
        self.components_dict["dd_conf_b"] = dd_conf_b 
        self.components_dict["dd_dilation_factor_b"] = dd_dilation_factor_b 
        self.components_dict["dd_offset_x_b"] = dd_offset_x_b 
        self.components_dict["dd_offset_y_b"] = dd_offset_y_b 
        self.components_dict["dd_mask_blur"] = dd_mask_blur 
        self.components_dict["dd_denoising_strength"] = dd_denoising_strength 
        self.components_dict["dd_random_seed"] = dd_random_seed 
        self.components_dict["dd_inpaint_full_res"] = dd_inpaint_full_res 
        self.components_dict["dd_inpaint_full_res_padding"] = dd_inpaint_full_res_padding 
        
        self.components_list = list(self.components_dict.values())

        #===============================
        def button_load_preset_click(v):
            return self.ConfigLoad(v)            

        button_load_preset.click( 
            fn=button_load_preset_click, 
            inputs=[selected_preset], outputs=self.components_list
        )
        
        button_save_preset.click( 
            fn=self.ConfigSave, 
            _js="function ask_(_, ...args){ var rlt = prompt('Save filename'); args.unshift(rlt); return args; }",
            inputs=[dummy_component, *self.components_list], outputs=[] 
        )
        #===============================
        def change_target_size_type(scale_index):
            is_custom_size = scale_index == 1
            is_custom_scale = scale_index == 2

            return [gr.update(visible=is_custom_size),
                    gr.update(visible=is_custom_size),
                    gr.update(visible=is_custom_scale)]

        target_size_type.change(
            fn=change_target_size_type,
            inputs=target_size_type,
            outputs=[custom_width, custom_height, custom_scale]
        )
        #===============================        
        def change_redraw_views(v_redraw_mode , v_save_image_hires_fix, v_redraw_on_hires_fix):
            redraw_mode_enum = GIDMode(v_redraw_mode)                
            b_is_redraw = redraw_mode_enum == GIDMode.CHESS or redraw_mode_enum == GIDMode.LINEAR
            v_hires = (v_save_image_hires_fix or (b_is_redraw and v_redraw_on_hires_fix))

            return [gr.update(visible=v_hires), gr.update(visible=v_hires), gr.update(visible=v_hires), gr.update(visible=v_hires),
                    gr.update(visible=b_is_redraw), gr.update(visible=b_is_redraw), gr.update(visible=b_is_redraw), gr.update(visible=b_is_redraw),
                    gr.update(visible=b_is_redraw), gr.update(visible=b_is_redraw), gr.update(visible=b_is_redraw), gr.update(visible=b_is_redraw),
                    gr.update(visible=b_is_redraw), gr.update(visible=b_is_redraw), gr.update(visible=b_is_redraw), gr.update(visible=b_is_redraw),
                    gr.update(visible=b_is_redraw)]
        
        temp_cps_in_hires_n_redraw = [redraw_mode,save_image_hires_fix, redraw_on_hires_fix]
        temp_cps_out_hires_n_redraw = [hiresfix_denoise, hiresfix_random_seed, hiresfix_steps, hiresfix_cfg, 
                     save_image_detailed, redraw_denoise, redraw_random_seed, redraw_steps, redraw_cfg, redraw_on_hires_fix, redraw_full_res, tile_width, tile_height, mask_blur, padding, show_redraw_steps]

        redraw_mode.change( fn=change_redraw_views, inputs=temp_cps_in_hires_n_redraw, outputs=temp_cps_out_hires_n_redraw )
        save_image_hires_fix.change( fn=change_redraw_views, inputs=temp_cps_in_hires_n_redraw, outputs=temp_cps_out_hires_n_redraw )
        redraw_on_hires_fix.change( fn=change_redraw_views, inputs=temp_cps_in_hires_n_redraw, outputs=temp_cps_out_hires_n_redraw )
        #===============================
        def change_use_partial_prompt(v):
            return [gr.update(visible=v), gr.update(visible=v), gr.update(visible=v), gr.update(visible=v), gr.update(visible=v), gr.update(visible=v)]

        use_partial_prompt.change(
            fn=change_use_partial_prompt, inputs=use_partial_prompt,
            outputs=[use_only_inputed_prompt, interrogator_name, interrogator_threshold, interrogator_max_token, interrogator_add_tags, interrogator_exclude_tags ])
        #===============================
        def select_fix_type(fix_index):
            all_visible = fix_index != 0
            mask_blur_visible = fix_index == 2 or fix_index == 3
            width_visible = fix_index == 1

            return [gr.update(visible=all_visible), gr.update(visible=all_visible), gr.update(visible=width_visible), gr.update(visible=mask_blur_visible), gr.update(visible=all_visible)]

        seams_fix_type.change(
            fn=select_fix_type, inputs=seams_fix_type,
            outputs=[save_image_seams_fix, seams_fix_denoise, seams_fix_width, seams_fix_mask_blur, seams_fix_padding]
        )

        #===============================

        return self.components_list

    def DisableAlwaysOnScript( self, p , filename ):
        scripts = getattr( p, "scripts", None)
        if scripts is not None:
            alwayson_scripts = getattr(scripts, "alwayson_scripts", None)
            if alwayson_scripts is not None:
                for idx, asc in enumerate(alwayson_scripts):                                
                    s = str(asc)
                    if filename in s:                                
                        alwayson_scripts.remove(asc)
                        return idx , asc
        return -1, None
    def RestoreAlwaysOnScript( self, p, idx , asc ):
        if asc is not None and idx >= 0:
            scripts = getattr(p, "scripts", None)
            if scripts is not None:
                alwayson_scripts = getattr(scripts, "alwayson_scripts", None)
                if alwayson_scripts is not None:
                    alwayson_scripts.insert(idx, asc)
        return -1, None

    def run(self, p,    
            disable_controlnet_during_detailup, show_redraw_steps,
            target_size_type, custom_width, custom_height, custom_scale, upscaler_name, save_image_upscale,
            save_image_hires_fix, hiresfix_denoise, hiresfix_random_seed, hiresfix_steps, hiresfix_cfg,
            redraw_mode, redraw_denoise, redraw_random_seed, redraw_steps, redraw_cfg, redraw_on_hires_fix, redraw_full_res, tile_width, tile_height, mask_blur, padding, 
            use_partial_prompt, use_only_inputed_prompt, interrogator_name, interrogator_threshold, interrogator_max_token, interrogator_add_tags, interrogator_exclude_tags,
            seams_fix_type, seams_fix_denoise, seams_fix_width, seams_fix_mask_blur, seams_fix_padding,
            save_image_detailed, save_image_seams_fix, 
            save_image_ddetailer, dd_model_a, dd_conf_a, dd_dilation_factor_a, dd_offset_x_a, dd_offset_y_a, dd_preprocess_b, dd_bitwise_op, dd_model_b, dd_conf_b, dd_dilation_factor_b, dd_offset_x_b, dd_offset_y_b, dd_mask_blur, dd_denoising_strength, dd_random_seed, dd_inpaint_full_res, dd_inpaint_full_res_padding):

        org_p = p
        org_seed = org_p.seed
        org_subseed = org_p.subseed
        org_n_iter = org_p.n_iter
        org_steps = org_p.steps
        org_cfg_scale = org_p.cfg_scale

        #Init        
        devices.torch_gc()        

        # Prepare Original image
        is_txt2img = isinstance(org_p, StableDiffusionProcessingTxt2Img)
        org_p.do_not_save_grid = True
        org_p.do_not_save_samples = True           
        org_p.batch_size = 1        
        org_p.n_iter = 1
        

        if is_txt2img:
            pass
        else:
            original_img = org_p.init_images[0]
            if original_img == None:
                return Processed(org_p, [], seed, "Empty image")
            original_img = images.flatten(original_img, opts.img2img_background_color)
            org_p.inpaint_full_res = False
            org_p.inpainting_fill = 1     

        seed = org_seed
        state.job_count = org_n_iter

        result_images = []

        tagger = GIDTagger() if use_partial_prompt else None
        if tagger is not None:
            i_interrogator_max_token = 0
            try:
                i_interrogator_max_token = int(interrogator_max_token)
            except:
                pass
            tagger.setup( use_only_inputed_prompt, interrogator_name, interrogator_threshold , interrogator_add_tags.rstrip(), interrogator_exclude_tags, i_interrogator_max_token )

        #S. For n_iter ===================================================================
        for n in range(org_n_iter):
            if state.interrupted:
                    break            
            devices.torch_gc()
            
            #Restore original seed, Fix seed
            org_p.seed = org_seed
            org_p.subseed = org_subseed
            processing.fix_seed(org_p)

            #S. Run t2i or i2i by default ==================================================
            if is_txt2img:
                state.job = "Draw t2i"
                print(f"Draw t2i. Prompt={org_p.prompt}")
                processed_org = processing.process_images(org_p)
                if (len(processed_org.images) > 0):
                    init_img = processed_org.images[0]
                
                p2 = StableDiffusionProcessingImg2Img(
                    init_images = None, resize_mode = 0, mask = None,
                    mask_blur= mask_blur,
                    inpainting_fill = 1, inpaint_full_res = True,
                    inpaint_full_res_padding= padding, inpainting_mask_invert= 0
                )                
                CopyAtoB( org_p, p2 ) #Copy All setting from original setting
            else:                
                state.job = "Draw i2i"
                print(f"Draw i2i. Prompt={org_p.prompt}")
                processed_org = processing.process_images(org_p)
                if (len(processed_org.images) > 0):
                    init_img = processed_org.images[0]
                p2 = copy.copy(org_p) #Copy All setting from original setting
            #E. Run t2i or i2i by default ==================================================

            if state.interrupted:
                    break
            
            #S. Prepare detailer ==================================================
            #override size
            if target_size_type == 1:
                p2.width = custom_width
                p2.height = custom_height
            if target_size_type == 2:
                p2.width = math.ceil((init_img.width * custom_scale) / 64) * 64
                p2.height = math.ceil((init_img.height * custom_scale) / 64) * 64

            
            if hasattr(p2 , "control_net_enabled"):
                print(f"control_net_enabled={p2.control_net_enabled}")

            # Upscaling
            state.job = "Upscaling"
            upscaler = GIDUpscaler(p2, init_img, upscaler_name, tile_width, tile_height)
            #Add upscaled image to list
            if show_redraw_steps:
                result_images = processed_org.images + result_images
            upscaler.upscale()

            if show_redraw_steps or save_image_upscale:
                result_images.insert(0, upscaler.image )
            if save_image_upscale:
                upscaler.save_image( processed_org.infotext(org_p, 0), suffix="(upscale)" )
            
            redraw_mode_enum = GIDMode(redraw_mode)
            upscaler.setup_redraw(redraw_mode_enum, redraw_full_res, padding, mask_blur, redraw_denoise, redraw_random_seed, redraw_steps, redraw_cfg, show_redraw_steps, 
                                  use_partial_prompt, tagger)
            upscaler.setup_seams_fix(seams_fix_padding, seams_fix_denoise, seams_fix_mask_blur, seams_fix_width, seams_fix_type)
            upscaler.print_info()                
            
            if n == 0:
                state.job_count += ( upscaler.calc_jobs_count() * org_n_iter )            
            #E. Prepare detailer ==================================================

            if state.interrupted:
                break

            #S. Hires. Fix===============================
            if save_image_hires_fix or redraw_on_hires_fix:   
                state.job = "Hires. fix"
                if n == 0:
                    state.job_count += ( 1 * org_n_iter )

                print("Generate Hires. fix image")               
                p_hires_fix = copy.copy(p2)
                if disable_controlnet_during_detailup:
                    sc_controlnet_idx, sc_controlnet = self.DisableAlwaysOnScript( p_hires_fix, "controlnet.py")

                if hiresfix_random_seed:
                    p_hires_fix.seed = -1
                    p_hires_fix.subseed = -1
                    processing.fix_seed(p_hires_fix)

                if hiresfix_steps > 0:
                    p_hires_fix.steps = hiresfix_steps
                if hiresfix_cfg > 0:
                    p_hires_fix.cfg_scale = hiresfix_cfg

                p_hires_fix.inpaint_full_res = True
                p_hires_fix.width = upscaler.image.width
                p_hires_fix.height = upscaler.image.height
                p_hires_fix.denoising_strength = hiresfix_denoise
                p_hires_fix.init_images = [ upscaler.image ]

                
                processed_hires_fix = processing.process_images(p_hires_fix)
                image_hires_fix = processed_hires_fix.images[0]

                if save_image_hires_fix:
                    p_hires_fix.extra_generation_params["\n[ Gix Detailer Hires. fix ]"] = "\n" + \
                        f"upscaler : {upscaler.upscaler_name}, " + \
                        f"denoise : {hiresfix_denoise}, " + \
                        f"steps : {p_hires_fix.steps}, " + \
                        f"cfg : {p_hires_fix.cfg_scale}, " + \
                        f"seed : {p_hires_fix.seed}, subseed : {p_hires_fix.subseed} "                
                    upscaler.save_image( processed_hires_fix.infotext(p_hires_fix, 0), image_hires_fix, suffix="(hires)" )
                    result_images = processed_hires_fix.images + result_images
                if redraw_on_hires_fix:
                    upscaler.image = image_hires_fix
                
                if disable_controlnet_during_detailup:
                    sc_controlnet_idx, sc_controlnet = self.RestoreAlwaysOnScript( p_hires_fix, sc_controlnet_idx, sc_controlnet )
            #E. Hires. Fix===============================
            if state.interrupted:
                break
            
            
            sc_controlnet_idx = -1
            sc_controlnet = None
            
            #S. Redraw===============================
            if upscaler.redraw.enabled:
                state.job = "Redraw detailing"

                if disable_controlnet_during_detailup:
                    sc_controlnet_idx, sc_controlnet = self.DisableAlwaysOnScript( p2, "controlnet.py")

                upscaler.image = upscaler.redraw.start(upscaler.p, upscaler.image, upscaler.rows, upscaler.cols, result_images)
                #self.initial_info = self.redraw.initial_info
                if save_image_detailed or show_redraw_steps:
                    result_images.insert(0, upscaler.image)
                
                if save_image_detailed:
                    s_ex_gen_p_log = "\n" + \
                        f"denoise : {upscaler.redraw.redraw_denoise}, " + \
                        f"sampling steps : {upscaler.redraw.redraw_steps}, " + \
                        f"cfg : {upscaler.redraw.redraw_cfg}, " + \
                        f"tile_size : {upscaler.redraw.tile_width}x{upscaler.redraw.tile_height}, " + \
                        f"mask_blur : {upscaler.p.mask_blur}, " + \
                        f"padding : {upscaler.redraw.padding}, " + \
                        f"use_partial_prompt : {upscaler.redraw.use_partial_prompt}"
                    
                    if upscaler.redraw.use_partial_prompt:
                        s_ex_gen_p_log = s_ex_gen_p_log + ", " + \
                        f"use_only_inputed_prompt : {upscaler.redraw.use_only_inputed_prompt}, " + \
                        f"interrogator_name : {interrogator_name}, " + \
                        f"interrogator_threshold : {interrogator_threshold}, " + \
                        f"interrogator_max_token : {interrogator_max_token}, " + \
                        f"interrogator_add_tags : {interrogator_add_tags}, " + \
                        f"interrogator_exclude_tags : {interrogator_exclude_tags}"

                    p2.extra_generation_params["\n[ Gix Detailer - Redraw ]"] = s_ex_gen_p_log
                    
                    if upscaler.redraw.logs is not None:
                        p2.extra_generation_params["\n[ Gix Detailer - Redraw logs ]"] = "\n" + "\n".join(upscaler.redraw.logs) + "\n"
                                    
                    initial_info_redraw = processed_org.infotext(p2, 0)
                    upscaler.save_image(initial_info_redraw, suffix="(detailer)")
                
                if disable_controlnet_during_detailup:
                    sc_controlnet_idx, sc_controlnet = self.RestoreAlwaysOnScript( p2, sc_controlnet_idx, sc_controlnet )
            #E. Redraw===============================
            if state.interrupted:
                break
            #S. Seams Fix===============================
            if upscaler.seams_fix.enabled:
                state.job = "Seams fix"

                if disable_controlnet_during_detailup:
                    sc_controlnet_idx, sc_controlnet = self.DisableAlwaysOnScript( p2, "controlnet.py")

                upscaler.image = upscaler.seams_fix.start(p2, upscaler.image, upscaler.rows, upscaler.cols)
                
                #self.result_images.insert(0, self.image)
                if save_image_seams_fix:
                    initial_info_seams = processed_org.infotext(p2, 0)                    
                    upscaler.save_image(initial_info_seams, suffix="(seams)")
                    result_images.insert(0, upscaler.image)

                if disable_controlnet_during_detailup:
                    sc_controlnet_idx, sc_controlnet = self.RestoreAlwaysOnScript( p2, sc_controlnet_idx, sc_controlnet )
            #E. Seams Fix===============================

            #S. Detection Detailer =====================
            if dd_model_a != "None": 
                state.job = "Detection detailing"

                dd_image = upscaler.image

                p_dd = copy.copy(p2)
                p_dd.width = dd_image.width
                p_dd.height = dd_image.height
                p_dd.init_images = None
                p_dd.resize_mode = 0
                p_dd.denoising_strength = dd_denoising_strength
                p_dd.mask = None
                p_dd.mask_blur= dd_mask_blur
                p_dd.inpainting_fill = 1
                p_dd.inpaint_full_res = dd_inpaint_full_res
                p_dd.inpaint_full_res_padding= dd_inpaint_full_res_padding
                p_dd.inpainting_mask_invert= 0

                if disable_controlnet_during_detailup:
                    sc_controlnet_idx, sc_controlnet = self.DisableAlwaysOnScript( p_dd, "controlnet.py")
                
                processing.fix_seed(p_dd)
                
                processed_dd = None
                masks_a = []
                masks_b_pre = []
                logs_dd = []
                # Optional secondary pre-processing run
                if (dd_model_b != "None" and dd_preprocess_b): 
                    label_b_pre = "B"
                    results_b_pre = dd_inference(dd_image, dd_model_b, dd_conf_b/100.0, label_b_pre)
                    masks_b_pre = dd_create_segmasks(results_b_pre)
                    masks_b_pre = dd_dilate_masks(masks_b_pre, dd_dilation_factor_b, 1)
                    masks_b_pre = dd_offset_masks(masks_b_pre,dd_offset_x_b, dd_offset_y_b)
                    if (len(masks_b_pre) > 0):
                        results_b_pre = dd_update_result_masks(results_b_pre, masks_b_pre)
                        if show_redraw_steps:
                            segmask_preview_b = dd_create_segmask_preview(results_b_pre, dd_image)
                            result_images.insert(0, segmask_preview_b)
                        #shared.state.current_image = segmask_preview_b
                        gen_count = len(masks_b_pre)
                        state.job_count += gen_count
                        print(f"Processing {gen_count} model {label_b_pre} detections for output generation {n + 1}.")
                        p_dd.init_images = [dd_image]
                        
                        for i in range(gen_count):
                            p_dd.image_mask = masks_b_pre[i]                                  
                            crop_region = masking.get_crop_region(np.array(p_dd.image_mask), p_dd.inpaint_full_res_padding)
                            crop_region_list = list(crop_region) #tuple cannot item assignment
                            crop_region_list[0] += dd_inpaint_full_res_padding
                            crop_region_list[1] += dd_inpaint_full_res_padding
                            crop_region_list[2] -= dd_inpaint_full_res_padding
                            crop_region_list[3] -= dd_inpaint_full_res_padding
                            crop_region = tuple(crop_region_list)
                            if use_partial_prompt:                                                                
                                image_sliced = dd_image.crop(crop_region)
                                p_dd.prompt = tagger.GetPartialPrompts( image_sliced, org_p.prompt )
                                image_sliced = None

                            if dd_random_seed:
                                p_dd.seed = -1
                                p_dd.subseed = -1
                                processing.fix_seed(p_dd)
                                
                            processed_dd = processing.process_images(p_dd)
                            p_dd.seed = processed_dd.seed + 1
                            p_dd.init_images = processed_dd.images

                            s_log = f"D-Detailer. Rect (x{crop_region[0]} y{crop_region[1]}) to (x{crop_region[2]}, y{crop_region[3]}),\n" + \
                                f"seed : {p_dd.seed} , subseed : {p_dd.subseed}, steps : {p_dd.steps}, cfg : {p_dd.cfg_scale}  \n" + \
                                f"Prompt={p_dd.prompt}"
                            logs_dd.append(s_log)
                        if (gen_count > 0):                            
                            dd_image = processed_dd.images[0]
                    else:
                        print(f"No model B detections for output generation {n} with current settings.")

                # Primary run
                if (dd_model_a != "None"):
                    label_a = "A"
                    if (dd_model_b != "None" and dd_bitwise_op != "None"):
                        label_a = dd_bitwise_op
                    results_a = dd_inference(dd_image, dd_model_a, dd_conf_a/100.0, label_a)
                    masks_a = dd_create_segmasks(results_a)
                    masks_a = dd_dilate_masks(masks_a, dd_dilation_factor_a, 1)
                    masks_a = dd_offset_masks(masks_a,dd_offset_x_a, dd_offset_y_a)
                    if (dd_model_b != "None" and dd_bitwise_op != "None"):
                        label_b = "B"
                        results_b = dd_inference(dd_image, dd_model_b, dd_conf_b/100.0, label_b)
                        masks_b = dd_create_segmasks(results_b)
                        masks_b = dd_dilate_masks(masks_b, dd_dilation_factor_b, 1)
                        masks_b = dd_offset_masks(masks_b,dd_offset_x_b, dd_offset_y_b)
                        if (len(masks_b) > 0):
                            combined_mask_b = dd_combine_masks(masks_b)
                            for i in reversed(range(len(masks_a))):
                                if (dd_bitwise_op == "A&B"):
                                    masks_a[i] = dd_bitwise_and_masks(masks_a[i], combined_mask_b)
                                elif (dd_bitwise_op == "A-B"):
                                    masks_a[i] = dd_subtract_masks(masks_a[i], combined_mask_b)
                                if (dd_is_allblack(masks_a[i])):
                                    del masks_a[i]
                                    for result in results_a:
                                        del result[i]                                        
                        else:
                            print("No model B detections to overlap with model A masks")
                            results_a = []
                            masks_a = []
                    
                    if (len(masks_a) > 0):
                        results_a = dd_update_result_masks(results_a, masks_a)
                        if show_redraw_steps:
                            segmask_preview_a = dd_create_segmask_preview(results_a, dd_image)
                            result_images.insert(0, segmask_preview_a)
                        #shared.state.current_image = segmask_preview_a
                        gen_count = len(masks_a)
                        state.job_count += gen_count
                        print(f"Processing {gen_count} model {label_a} detections for output generation {n + 1}.")                    
                        p_dd.init_images = [dd_image]

                        for i in range(gen_count):
                            p_dd.image_mask = masks_a[i]      

                            crop_region = masking.get_crop_region(np.array(p_dd.image_mask), p_dd.inpaint_full_res_padding)
                            crop_region_list = list(crop_region) #tuple cannot item assignment
                            crop_region_list[0] += dd_inpaint_full_res_padding
                            crop_region_list[1] += dd_inpaint_full_res_padding
                            crop_region_list[2] -= dd_inpaint_full_res_padding
                            crop_region_list[3] -= dd_inpaint_full_res_padding
                            crop_region = tuple(crop_region_list)
                            if use_partial_prompt:                                
                                image_sliced = dd_image.crop(crop_region)
                                p_dd.prompt = tagger.GetPartialPrompts( image_sliced, org_p.prompt )
                                image_sliced = None
                            
                            if dd_random_seed:
                                p_dd.seed = -1
                                p_dd.subseed = -1
                                processing.fix_seed(p_dd)

                            processed_dd = processing.process_images(p_dd)
                            p_dd.seed = processed_dd.seed + 1
                            p_dd.init_images = processed_dd.images

                            s_log = f"D-Detailer. Rect (x{crop_region[0]} y{crop_region[1]}) to (x{crop_region[2]}, y{crop_region[3]}),\n" + \
                                f"seed : {p_dd.seed} , subseed : {p_dd.subseed}, steps : {p_dd.steps}, cfg : {p_dd.cfg_scale}  \n" + \
                                f"Prompt={p_dd.prompt}"
                            logs_dd.append(s_log)
                        if (gen_count > 0):
                            dd_image = processed_dd.images[0]                        
                    else: 
                        print(f"No model {label_a} detections for output generation {n} with current settings.")
                
                if save_image_ddetailer:     
                    if processed_dd is None:
                        print( "No detections found." )
                    else:
                        p_dd.extra_generation_params["\n[ Detection detailer logs ]"] = "\n" + "\n".join(logs_dd) + "\n"
                    initial_info_dd = processed_org.infotext(p_dd, 0) 
                    upscaler.save_image(initial_info_dd, dd_image, suffix="(ddetailer)")
                    result_images.insert(0, dd_image)

                if disable_controlnet_during_detailup:
                    sc_controlnet_idx, sc_controlnet = self.RestoreAlwaysOnScript( p_dd, sc_controlnet_idx, sc_controlnet )
                    
            #E. Detection Detailer =====================
            
        #E. For n_iter ===================================================================

        #===========================================================
        initial_info_org = processed_org.infotext(org_p, 0)
        
        gc.collect()
        devices.torch_gc()

        return Processed(org_p, result_images, seed, initial_info_org)

#E. Detailer ===================================================

