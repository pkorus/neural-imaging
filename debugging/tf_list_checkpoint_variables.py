import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def usage():
    print('python3 tf_list_checkpoint_variables.py <model directory>')

def main(argv):

    if len(argv) != 1:
        usage()
        sys.exit(0)

    checkpoint_dir = argv[0]

    if not os.path.exists(checkpoint_dir):
        print('Directory does not exist!')
        sys.exit(1)

    import tensorflow as tf

    replace_from = None
    replace_to = None
    dry_run = True
    add_prefix = False

    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)

    print('Checkpoints:\n', checkpoint)

    with tf.Session() as sess:
        for i, (var_name, _) in enumerate(tf.contrib.framework.list_variables(checkpoint_dir)):
            # Load the variable
            var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)

            print('{0:3d}.  {1:30s} -> {2.shape}'.format(i, var_name, var))

        #     # Set the new name
        #     new_name = var_name
        #     if None not in [replace_from, replace_to]:
        #         new_name = new_name.replace(replace_from, replace_to)
        #     if add_pref ix:
        #         new_name = add_prefix + new_name

        #     if dry_run:
        #         print('%30s -> %s.' % (var_name, new_name))
        #     else:
        #         print('Renaming %s to %s.' % (var_name, new_name))
        #         # Rename the variable
        #         var = tf.Variable(var, name=new_name)

        # if not dry_run:
        #     # Save the variables
        #     saver = tf.train.Saver()
        #     sess.run(tf.global_variables_initializer())
        #     saver.save(sess, checkpoint.model_checkpoint_path)


if __name__ == '__main__':
    main(sys.argv[1:])
