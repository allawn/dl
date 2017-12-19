import tensorflow as tf
slim = tf.contrib.slim

ckpt="vgg_19.ckpt"

variables_to_restore = []
variables_to_restore_all = []
numa=1
for var in slim.get_model_variables():
    print(numa,var.op.name)
    numa=numa+1
    if not var.op.name.startswith('vgg_19/fc') and not var.op.name.startswith('global_step'):
        variables_to_restore.append(var)
    if not var.op.name.startswith('global_step'):
        variables_to_restore_all.append(var)

print("-----------var in check---------")

ckpt_reader = tf.train.NewCheckpointReader(ckpt)
ckpt_vars = ckpt_reader.get_variable_to_shape_map().keys()
vars_in_ckpt = {}
numa=1
for variable_name in sorted(ckpt_vars):
    print(numa,variable_name)
    numa=numa+1