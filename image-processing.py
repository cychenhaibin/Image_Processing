import tensorflow as tf
import tensorflow.compat.v1.logging as logging
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.utils import to_categorical
# 在代码的开头定义全局变量来存储ImageTk.PhotoImage对象
global_photo = None

# 移除TensorFlow的warnings
logging.set_verbosity(logging.ERROR)

# 解决中文乱码问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载Fashion MNIST数据集
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 将图片数据标准化
train_images = train_images / 255.0
test_images = test_images / 255.0

class_names = ['T恤', '裤子', '套头衫', '连衣裙', '外套',
                '凉鞋', '衬衫', '运动鞋', '包', '短靴']

# 展示部分样本图片
def display_sample_images(images, labels, num_rows=5, num_cols=10, figsize=(15, 7)):
    plt.figure(figsize=figsize)
    for i in range(num_rows * num_cols):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])
    plt.tight_layout()
    plt.show()
display_sample_images(train_images, train_labels)

# 一系列数据增强方法
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# 创建VGG模型
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')])
model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy'])

# 训练模型并储存历史数据
history = model.fit(train_images, train_labels, epochs=10, validation_split=0.2)

# 基于测试集评估数据
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nVGG模型准确率:', test_acc)
basic_model_predictions = model.predict(test_images)
basic_misclassified_indices = np.where(np.argmax(basic_model_predictions, axis=1) != test_labels)[0]
print(f"VGG模型错误分类的图像数目: {len(basic_misclassified_indices)}")

# 建立用于预测的数据集
train_predictions = model.predict(train_images)
train_pred_labels = np.argmax(train_predictions, axis=1)

# 计算分类错误的图像数目
misclassified_train = np.sum(train_pred_labels != train_labels)

# 可视化训练结果（界面）
def training_vis(hist, misclassified):
    # 提取训练和验证损失和准确性
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']

    # 画图
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # 损失的子图
    axs[0].plot(loss, label='训练集Loss')
    axs[0].plot(val_loss, label='验证集Loss')
    axs[0].set_xlabel('迭代周期')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('训练集和验证集的Loss值')
    axs[0].legend()
    axs[0].text(0.5, 0.5, f'错误分类图像数目: {misclassified}', transform=axs[0].transAxes, ha="center")

    # 准确性的子图
    axs[1].plot(acc, label='训练集准确度')
    axs[1].plot(val_acc, label='验证集准确度')
    axs[1].set_xlabel('迭代周期')
    axs[1].set_ylabel('准确度')
    axs[1].set_title('训练集和验证集的Loss值的准确度')
    axs[1].legend()
    plt.tight_layout()

 # 将训练历史可视化
training_vis(history, len(basic_misclassified_indices))

# 重构图像
train_images_reshaped = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images_reshaped = test_images.reshape((test_images.shape[0], 28, 28, 1))

# 第一次构建的CNN模型
cnn_model = tf.keras.Sequential([
    tf.keras.layers.Reshape(target_shape=(28, 28, 1), input_shape=(28, 28)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

cnn_model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

# 用训练集训练CNN模型
cnn_history = cnn_model.fit(datagen.flow(train_images_reshaped, train_labels, batch_size=32),
                            epochs=10,
                            validation_data=(test_images_reshaped, test_labels))
# 通过测试结果评价CNN模型
test_loss, test_acc = cnn_model.evaluate(test_images_reshaped, test_labels, verbose=2)
print('\nCNN模型准确率:', test_acc)
cnn_model_predictions = cnn_model.predict(test_images_reshaped)
cnn_misclassified_indices = np.where(np.argmax(cnn_model_predictions, axis=1) != test_labels)[0]
print(f"CNN模型错误分类的图像数目: {len(cnn_misclassified_indices)}")

training_vis(cnn_history, len(cnn_misclassified_indices))

# 修改后的CNN模型（降低层数）
cnn_model = tf.keras.Sequential([
    tf.keras.layers.Reshape(target_shape=(28, 28, 1), input_shape=(28, 28)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

cnn_model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

cnn_history = cnn_model.fit(train_images, train_labels, epochs=10, validation_split=0.2)

# 评价修改后的CNN模型
test_loss, test_acc = cnn_model.evaluate(test_images, test_labels, verbose=2)
print('\n修改后的CNN模型准确率:', test_acc)
cnn_model_predictions = cnn_model.predict(test_images)
cnn_misclassified_indices = np.where(np.argmax(cnn_model_predictions, axis=1) != test_labels)[0]
print(f"修改后的CNN模型错误分类的图像数目: {len(cnn_misclassified_indices)}")

training_vis(cnn_history, len(cnn_misclassified_indices))

# 定义可预测的概率模型
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

# 对训练集做出预测
predictions = probability_model.predict(test_images)

# 绘制样品预测类别的图像
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],100 * np.max(predictions_array),class_names[true_label]),color=color)

# 绘制图像预测结果
def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# 绘制带预测结果的测试图像
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# 绘制单一图像的预测结果的图像
i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

# 将标签转换为独热编码
y_test = to_categorical(test_labels, num_classes=len(class_names))

# 使用VGG和CNN两个模型预测每个类别的概率
y_score_basic = model.predict(test_images)
y_score_cnn = cnn_model.predict(test_images)

# 计算每个类别的 ROC 曲线和 AUC 的函数
def compute_roc_auc_per_class(y_test, y_score, n_classes):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return fpr, tpr, roc_auc

# 计算VGG和CNN两个模型每个类别的 ROC 曲线和 ROC 面积
fpr_basic, tpr_basic, roc_auc_basic = compute_roc_auc_per_class(y_test, y_score_basic, len(class_names))
fpr_cnn, tpr_cnn, roc_auc_cnn = compute_roc_auc_per_class(y_test, y_score_cnn, len(class_names))

# 绘制每个类别的ROC曲线
plt.figure(figsize=(15, 10))
colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
for i, color in zip(range(len(class_names)), colors):
    plt.plot(fpr_basic[i], tpr_basic[i], color=color, lw=2,
             label=f'VGG - {class_names[i]} (面积 = {roc_auc_basic[i]:0.2f})')
    plt.plot(fpr_cnn[i], tpr_cnn[i], color=color, lw=2, linestyle='--',
             label=f'CNN - {class_names[i]} (面积 = {roc_auc_cnn[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('每一个种类的ROC曲线')
plt.legend(loc="lower right")
plt.show()
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from PIL import Image, ImageTk

# Your existing code here...

# Function to display misclassified images
def display_misclassified_images(model, test_images, test_labels, class_names):
    global global_photo  # 引用全局变量

    predictions = model.predict(test_images)
    misclassified_indices = np.where(np.argmax(predictions, axis=1) != test_labels)[0]

    root = tk.Tk()
    root.title("Misclassified Images")

    num_rows = 5
    num_cols = 5

    for i in range(min(num_rows * num_cols, len(misclassified_indices))):
        index = misclassified_indices[i]
        img = test_images[index]
        true_label = class_names[test_labels[index]]
        predicted_label = class_names[np.argmax(predictions[index])]

        label_text = f"True Label: {true_label}\nPredicted Label: {predicted_label}"

        frame = ttk.Frame(root)
        label = ttk.Label(frame, text=label_text)
        label.grid(row=0, column=0, padx=5, pady=5)

        img_label = ttk.Label(frame)
        img_label.grid(row=1, column=0, padx=5, pady=5)

        img *= 255
        img = img.astype(np.uint8)
        photo = ImageTk.PhotoImage(Image.fromarray(img))

        # 更新全局变量
        global_photo = photo

        img_label.configure(image=photo)
        img_label.image = photo

        frame.grid(row=i // num_cols, column=i % num_cols, padx=10, pady=10)

    root.mainloop()

# ...

# 在show_gui函数中
def show_gui(model, history, misclassified_indices):
    global global_photo  # 引用全局变量

    root = tk.Tk()
    root.title("Model Visualization")

    # Your existing code for training visualization here...

    # Button to display misclassified images
    misclassified_button = ttk.Button(root, text="Show Misclassified Images", command=lambda: display_misclassified_images(model, test_images, test_labels, class_names))
    misclassified_button.grid(row=2, column=0, pady=10)

    root.mainloop()

# Call the function to show the GUI
show_gui(model, history, basic_misclassified_indices)
