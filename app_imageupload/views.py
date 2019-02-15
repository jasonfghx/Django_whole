from django.shortcuts import render
import os#, sys
import tensorflow as tf

from .form import UploadImageForm
# 寫在 form.py 的內容: --------------------------------
# from django import forms
# class UploadImageForm(forms.Form):
#     """图像上传表单"""
#     #text = forms.CharField(max_length=100)
#     image = forms.ImageField(
#         label='上傳一張震測圖給石油王',
#     )
# ----------------------------------------------------
from .models import Image
# 寫在 models.py 的內容--------------------------------
# from django.db import models
# class Image(models.Model):
#     photo = models.ImageField(null=True, blank=True)
#     def __str__(self):
#         return self.photo.name
# ----------------------------------------------------
# source: https://github.com/ctudoudou/Keras-Web-demo ------------------
# ./app/view.py
from keras.models import load_model
import numpy as np

graph = tf.get_default_graph()
model = load_model('./2019-02-11T0735_.h5')     # 跟 manage.py 放同一層
# ----------------------------------------------------------------------

def index(request):
    """图片的上传"""
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            image = request.FILES['image']  # type = <class 'django.core.files.uploadedfile.InMemoryUploadedFile'>
            picture = Image(photo=request.FILES['image'])    # type = <class 'app_imageupload.models.Image'>
            picture.save()

            # 改寫:
            labels, scores = imageclassify(image)
            results = []
            for i in range(0, len(labels)):
                results.append(f'{labels[i]}: {scores[i]:.1%}')

            form = UploadImageForm()
            return render(request, 'show.html', {'picture': picture,
                                                 'results': results,
                                                 'form': form,
                                                 })

    else:
        form = UploadImageForm()
        return render(request, 'index.html', {'form': form})


def imageclassify(image):
    # source: https://github.com/ctudoudou/Keras-Web-demo
    # ./app/view.py
    import PIL
    global graph
    global model

    # result = ''
    # image = request.FILES['file']
    with open('dd.png', "wb+") as destination:
        for chunk in image.chunks():
            destination.write(chunk)
    # arr = np.array(Image.open('./dd.png').convert('L'))
    arr = np.array(PIL.Image.open('./dd.png').resize([224,224]))
    arr = arr /255.

    np.set_printoptions(formatter={'all':lambda x: str(round(x*100,1))+'%'})
    with graph.as_default():
        # result = f'預測结果：{model.predict(np.array([arr]))[0]}'
        predict_scores = model.predict(np.array([arr]))[0]

    labels = []
    with open(os.path.join(os.path.dirname(__file__), "labels.txt")) as label_file:
        for line in label_file.readlines():
            labels.append(line)

    return labels, predict_scores

def imageclassify_backup(picture):
    """将上传的图片进行图片识别分类"""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # image_path = picture

    # Read in the image_data
    image_data = tf.gfile.FastGFile(picture.photo.path, 'rb').read()

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line in tf.gfile.GFile("app_imageupload/retrained_labels.txt")]

    # Unpersists graph from file
    with tf.gfile.FastGFile("app_imageupload/retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        label = []
        i = 1
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            # print('%s (score = %.5f)' % (human_string, score))
            a = (human_string, score)
            if i < 2:
                label.append(a)
                i += 1
            else:
                break
        return label

