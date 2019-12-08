from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import numpy as np
import cv2

from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks
)
from yolov3_tf2.utils import freeze_all
import yolov3_tf2.dataset as dataset

flags.DEFINE_string("dataset", "./data/train.tfrecord", "path to dataset")
flags.DEFINE_string("val_dataset", "./data/val.tfrecord", "path to validation dataset")
flags.DEFINE_boolean("tiny", False, "yolov3 or yolov3-tiny")
flags.DEFINE_string("weights", "./checkpoints/yolov3.tf",
                    "path to weights file")
flags.DEFINE_string("classes", "./data/class.names", "path to classes file")
flags.DEFINE_enum("mode", "fit", ["fit", "eager_fit", "eager_tf","fit_multi_gpu"],
                  "fit: model.fit, "
                  "eager_fit: model.fit(run_eagerly=True), "
                  "eager_tf: custom GradientTape"
                  "fit_multi_gpu: use multiple gpus with model.fit")
flags.DEFINE_list("gpus",None,"Devices to use with mutli GPU training")
flags.DEFINE_enum("transfer", "none",
                  ["none", "darknet", "no_output", "frozen", "fine_tune"],
                  "none: Training from scratch, "
                  "darknet: Transfer darknet, "
                  "no_output: Transfer all but output, "
                  "frozen: Transfer and freeze all, "
                  "fine_tune: Transfer all and freeze darknet only")
flags.DEFINE_string("strategy","mirrored","Strategey for distubuted training")
flags.DEFINE_integer("size", 416, "image size")
flags.DEFINE_integer("epochs", 10, "number of epochs")
flags.DEFINE_integer("batch_size", 12, "batch size")
flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
flags.DEFINE_integer("num_classes", 1, "number of classes in the model")


def main(_argv):
    if FLAGS.mode != "fit_multi_gpu":
        if FLAGS.tiny:
            model = YoloV3Tiny(FLAGS.size, training=True,
                               classes=FLAGS.num_classes)
            anchors = yolo_tiny_anchors
            anchor_masks = yolo_tiny_anchor_masks
        else:
            model = YoloV3(FLAGS.size, training=True, classes=FLAGS.num_classes)
            anchors = yolo_anchors
            anchor_masks = yolo_anchor_masks

        train_dataset = dataset.load_fake_dataset()
        if FLAGS.dataset:
            train_dataset = dataset.load_tfrecord_dataset(
                FLAGS.dataset, FLAGS.classes, FLAGS.size)
        train_dataset = train_dataset.shuffle(buffer_size=1024)  # TODO: not 1024
        train_dataset = train_dataset.batch(FLAGS.batch_size)
        train_dataset = train_dataset.map(lambda x, y: (
            dataset.transform_images(x, FLAGS.size),
            dataset.transform_targets(y, anchors, anchor_masks, FLAGS.classes)))
        train_dataset = train_dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)

        val_dataset = dataset.load_fake_dataset()
        if FLAGS.val_dataset:
            val_dataset = dataset.load_tfrecord_dataset(
                FLAGS.val_dataset, FLAGS.classes, FLAGS.size)
        val_dataset = val_dataset.batch(FLAGS.batch_size)
        val_dataset = val_dataset.map(lambda x, y: (
            dataset.transform_images(x, FLAGS.size),
            dataset.transform_targets(y, anchors, anchor_masks, FLAGS.classes)))

        if FLAGS.transfer != "none":
            model.load_weights(FLAGS.weights)
            if FLAGS.transfer == "fine_tune":
                # freeze darknet
                darknet = model.get_layer("yolo_darknet")
                freeze_all(darknet)
            elif FLAGS.mode == "frozen":
                # freeze everything
                freeze_all(model)
            else:
                # reset top layers
                if FLAGS.tiny:  # get initial weights
                    init_model = YoloV3Tiny(
                        FLAGS.size, training=True, classes=FLAGS.num_classes)
                else:
                    init_model = YoloV3(
                        FLAGS.size, training=True, classes=FLAGS.num_classes)

                if FLAGS.transfer == "darknet":
                    for l in model.layers:
                        if l.name != "yolo_darknet" and l.name.startswith("yolo_"):
                            l.set_weights(init_model.get_layer(
                                l.name).get_weights())
                        else:
                            freeze_all(l)
                elif FLAGS.transfer == "no_output":
                    for l in model.layers:
                        if l.name.startswith("yolo_output"):
                            l.set_weights(init_model.get_layer(
                                l.name).get_weights())
                        else:
                            freeze_all(l)

        optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
        loss = [YoloLoss(anchors[mask], classes=FLAGS.num_classes)
                for mask in anchor_masks]

        if FLAGS.mode == "eager_tf" :
            # Eager mode is great for debugging
            # Non eager graph mode is recommended for real training
            avg_loss = tf.keras.metrics.Mean("loss", dtype=tf.float32)
            avg_val_loss = tf.keras.metrics.Mean("val_loss", dtype=tf.float32)

            for epoch in range(1, FLAGS.epochs + 1):
                for batch, (images, labels) in enumerate(train_dataset):
                    with tf.GradientTape() as tape:
                        outputs = model(images, training=True)
                        regularization_loss = tf.reduce_sum(model.losses)
                        pred_loss = []
                        for output, label, loss_fn in zip(outputs, labels, loss):
                            true_box, true_obj, true_class_idx = tf.split(
                                label, (4, 1, 1), axis=-1)
                            #print(np.nonzero(true_box.numpy))
                            #exit()
                            pred_loss.append(loss_fn(label, output))
                        total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                    grads = tape.gradient(total_loss, model.trainable_variables)
                    optimizer.apply_gradients(
                        zip(grads, model.trainable_variables))
                    print(tf.executing_eagerly())
                    logging.info("{}_train_{}, {}, {}".format(
                        epoch, batch, total_loss.numpy(),
                        list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                    avg_loss.update_state(total_loss)

                for batch, (images, labels) in enumerate(val_dataset):
                    outputs = model(images)
                    regularization_loss = tf.reduce_sum(model.losses)
                    pred_loss = []
                    for output, label, loss_fn in zip(outputs, labels, loss):
                        pred_loss.append(loss_fn(label, output))
                    total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                    logging.info("{}_val_{}, {}, {}".format(
                        epoch, batch, total_loss.numpy(),
                        list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                    avg_val_loss.update_state(total_loss)

                logging.info("{}, train: {}, val: {}".format(
                    epoch,
                    avg_loss.result().numpy(),
                    avg_val_loss.result().numpy()))

                avg_loss.reset_states()
                avg_val_loss.reset_states()
                model.save_weights(
                "checkpoints/yolov3_train_{}.tf".format(epoch))

        else:
            model.compile(optimizer=optimizer, loss=loss,
                        run_eagerly=(FLAGS.mode == "eager_fit"))
        
            checkpoint_dir = "checkpoints"
            log_dir = "logs"

            callbacks = [
                ReduceLROnPlateau(verbose=1),
                EarlyStopping(patience=3, verbose=1),
                ModelCheckpoint(checkpoint_dir + "/yolov3_train_{epoch}.ckpt",
                                verbose=1, save_weights_only=True),
                TensorBoard(log_dir=log_dir)]

            history = model.fit(train_dataset,
                        epochs=FLAGS.epochs,
                        callbacks=callbacks,
                        validation_data=val_dataset)
    else:
        if FLAGS.strategy.lower() == "mirrored":
            strategy = tf.distribute.MirroredStrategy(devices = FLAGS.gpus) 
        else:
            pass # TODO: Add other strategies
        n_devices = strategy.num_replicas_in_sync
        logging.info("Using {} devices:".format(n_devices))
        for dev in FLAGS.gpus if FLAGS.gpus else tf.config.experimental.list_physical_devices(device_type=None):
            logging.info("\t" + str(dev))


        GLOBAL_BATCH_SIZE = FLAGS.batch_size * n_devices

        with strategy.scope():
            if FLAGS.tiny:
                model = YoloV3Tiny(FLAGS.size, training=True,
                                    classes=FLAGS.num_classes)
                anchors = yolo_tiny_anchors
                anchor_masks = yolo_tiny_anchor_masks
            else:
                model = YoloV3(FLAGS.size, training=True, classes=FLAGS.num_classes)
                anchors = yolo_anchors
                anchor_masks = yolo_anchor_masks

            optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
            loss = [YoloLoss(anchors[mask], classes=FLAGS.num_classes)
                for mask in anchor_masks]

            model.compile(optimizer=optimizer, loss=loss)

        train_dataset = dataset.load_fake_dataset()
        if FLAGS.dataset:
            train_dataset = dataset.load_tfrecord_dataset(
                FLAGS.dataset, FLAGS.classes, FLAGS.size)
        train_dataset = train_dataset.shuffle(buffer_size=1024)  # TODO: not 1024
        train_dataset = train_dataset.batch(GLOBAL_BATCH_SIZE)
        train_dataset = train_dataset.map(lambda x, y: (
            dataset.transform_images(x, FLAGS.size),
            dataset.transform_targets(y, anchors, anchor_masks, FLAGS.classes)))
        train_dataset = train_dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)

        val_dataset = dataset.load_fake_dataset()
        if FLAGS.val_dataset:
            val_dataset = dataset.load_tfrecord_dataset(
                FLAGS.val_dataset, FLAGS.classes, FLAGS.size)
        val_dataset = val_dataset.batch(GLOBAL_BATCH_SIZE)
        val_dataset = val_dataset.map(lambda x, y: (
            dataset.transform_images(x, FLAGS.size),
            dataset.transform_targets(y, anchors, anchor_masks, FLAGS.classes)))
        
        checkpoint_dir = "checkpoints"
        log_dir = "logs"

        callbacks = [
            ReduceLROnPlateau(verbose=1),
            EarlyStopping(patience=3, verbose=1),
            ModelCheckpoint(checkpoint_dir + "/yolov3_train_{epoch}.ckpt",
                            verbose=1, save_weights_only=True),
            TensorBoard(log_dir=log_dir)]

        history = model.fit(train_dataset,
                    epochs=FLAGS.epochs,
                    callbacks=callbacks,
                    validation_data=val_dataset)
            
if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
    