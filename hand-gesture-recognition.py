{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "2Lx721HFR-bw"
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.metrics import classification_report, accuracy_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "Av85bNYmTiXE",
    "outputId": "ec100dcd-7fc0-4efa-eb97-eaee645d67f6"
   },
   "outputs": [],
   "source": [
    "!pip install opendatasets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "dfrLJOcDTjIv",
    "outputId": "5e14c8f5-ace2-423a-a0d2-d1462554cb08"
   },
   "outputs": [],
   "source": [
    "import opendatasets as od\n",
    "od.download(\n",
    "   \"https://www.kaggle.com/datasets/gti-upm/leapgestrecog/data\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "cTcXxaI4zjTe"
   },
   "outputs": [],
   "source": [
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "import keras\n",
    "import matplotlib.pyplot as plt\n",
    "import os\n",
    "import cv2\n",
    "import numpy as np\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "from keras.layers import Conv2D, Activation, MaxPool2D, Dense, Flatten, Dropout"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "YZtxD0yRzrca"
   },
   "outputs": [],
   "source": [
    "CATEGORIES = [\"01_palm\", '02_l','03_fist','04_fist_moved','05_thumb','06_index','07_ok','08_palm_moved','09_c','10_down']\n",
    "IMG_SIZE = 50\n",
    "data_path = \"/content/leapgestrecog/leapGestRecog\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "KZ3P0U0Pzytr",
    "outputId": "7d4b782e-564e-476c-fbfc-fd1ff0044e1a"
   },
   "outputs": [],
   "source": [
    "image_data = []\n",
    "for dr in os.listdir(data_path):\n",
    "    for category in CATEGORIES:\n",
    "        class_index = CATEGORIES.index(category)\n",
    "        path = os.path.join(data_path, dr, category)\n",
    "        for img in os.listdir(path):\n",
    "            try:\n",
    "                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)\n",
    "                image_data.append([cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE)), class_index])\n",
    "            except Exception as e:\n",
    "                pass\n",
    "image_data[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "_fXSaF-T0SlB"
   },
   "outputs": [],
   "source": [
    "import random\n",
    "random.shuffle(image_data)\n",
    "input_data = []\n",
    "label = []"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "FB61iFU7z18B",
    "outputId": "0359203f-2dcc-4595-e6ef-18f3f8ad3bb1"
   },
   "outputs": [],
   "source": [
    "for X, y in image_data:\n",
    "    input_data.append(X)\n",
    "    label.append(y)\n",
    "label[:10]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "pWNfCAUn0VA8",
    "outputId": "e0925149-1877-4f7a-ae19-38bfea3ba75f"
   },
   "outputs": [],
   "source": [
    "plt.figure(1, figsize=(10,10))\n",
    "for i in range(1,10):\n",
    "    plt.subplot(3,3,i)\n",
    "    plt.imshow(image_data[i][0], cmap='hot')\n",
    "    plt.xticks([])\n",
    "    plt.yticks([])\n",
    "    plt.title(CATEGORIES[label[i]][3:])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "5de36Odj0YsQ",
    "outputId": "4efdf6a5-8bd6-48d3-a797-01079fcd383f"
   },
   "outputs": [],
   "source": [
    "input_data = np.array(input_data)\n",
    "label = np.array(label)\n",
    "input_data = input_data/255.0\n",
    "input_data.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "kdaWAQv_0uYk",
    "outputId": "e783858b-5980-43fb-c5a5-a33d34be3452"
   },
   "outputs": [],
   "source": [
    "label = keras.utils.to_categorical(label, num_classes=10)\n",
    "label[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "lpLzzr3V0_RH"
   },
   "outputs": [],
   "source": [
    "input_data.shape = (-1, IMG_SIZE, IMG_SIZE, 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "Aiu7iHQT1Dff"
   },
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split(input_data, label, test_size = 0.3, random_state=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "6LV9ECMY1MWF"
   },
   "outputs": [],
   "source": [
    "model = keras.models.Sequential()\n",
    "\n",
    "model.add(Conv2D(filters = 32, kernel_size = (3,3), input_shape = (IMG_SIZE, IMG_SIZE, 1)))\n",
    "model.add(Activation('relu'))\n",
    "\n",
    "\n",
    "model.add(Conv2D(filters = 32, kernel_size = (3,3)))\n",
    "model.add(Activation('relu'))\n",
    "model.add(MaxPool2D(pool_size=(2,2)))\n",
    "model.add(Dropout(0.3))\n",
    "\n",
    "model.add(Conv2D(filters = 64, kernel_size = (3,3)))\n",
    "model.add(Activation('relu'))\n",
    "model.add(MaxPool2D(pool_size=(2,2)))\n",
    "model.add(Dropout(0.3))\n",
    "\n",
    "model.add(Flatten())\n",
    "model.add(Dense(256, activation='relu'))\n",
    "model.add(Dense(10, activation='softmax'))\n",
    "\n",
    "model.compile(loss='categorical_crossentropy',\n",
    "             optimizer = 'Adam',\n",
    "             metrics = ['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "6XMwdzbc1YNI",
    "outputId": "16fb097a-ee65-4a37-de6b-9b9ab5533db2"
   },
   "outputs": [],
   "source": [
    "model.fit(X_train, y_train, epochs = 7, batch_size=32, validation_data=(X_test, y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "4SxTj_mq1bmD",
    "outputId": "9acad2d8-e01e-4bf9-aa15-1669bc4f430a"
   },
   "outputs": [],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "ZyTJsar8142-",
    "outputId": "8862912d-c4bf-4b44-b8c9-75590fce530b"
   },
   "outputs": [],
   "source": [
    "plt.plot(model.history.history['loss'])\n",
    "plt.plot(model.history.history['val_loss'])\n",
    "plt.title('Model Loss')\n",
    "plt.ylabel('Loss')\n",
    "plt.xlabel('Epochs')\n",
    "plt.legend(['train', 'test'])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "qiL1fbhY2c90",
    "outputId": "43aa811d-49a7-41c2-ec95-168efec9a974"
   },
   "outputs": [],
   "source": [
    "plt.plot(model.history.history['accuracy'])\n",
    "plt.plot(model.history.history['val_accuracy'])\n",
    "plt.title('Model Accuracy')\n",
    "plt.ylabel('Accuracy')\n",
    "plt.xlabel('Epochs')\n",
    "plt.legend(['train', 'test'])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "hUh0P11N2gqS",
    "outputId": "54e6faf4-8b30-4ad7-b19e-236f3d1aed5f"
   },
   "outputs": [],
   "source": [
    "test_loss, test_accuracy = model.evaluate(X_test, y_test)\n",
    "print('Test accuracy: {:2.2f}%'.format(test_accuracy*100))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "djmP34PF2j3f",
    "outputId": "2a109b31-43b4-4513-dcda-6564a3289e49"
   },
   "outputs": [],
   "source": [
    "from sklearn.metrics import confusion_matrix\n",
    "import seaborn as sn\n",
    "cat = [c[3:] for c in CATEGORIES]\n",
    "plt.figure(figsize=(10,10))\n",
    "cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test), axis=1))\n",
    "sn.heatmap(cm, annot=True,xticklabels=cat, yticklabels=cat)\n",
    "plt.plot()"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "gpuType": "T4",
   "provenance": []
  },
  "kaggle": {
   "accelerator": "none",
   "dataSources": [
    {
     "datasetId": 39466,
     "sourceId": 61155,
     "sourceType": "datasetVersion"
    }
   ],
   "dockerImageVersionId": 30761,
   "isGpuEnabled": false,
   "isInternetEnabled": true,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
