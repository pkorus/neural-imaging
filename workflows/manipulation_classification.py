import numpy as np
import tensorflow as tf
from collections import OrderedDict

from compression import codec
from models import pipelines, jpeg, forensics
from helpers import tf_helpers

from loguru import logger


class ManipulationClassification(object):

    def __init__(self, nip_model, manipulations=None, distribution=None, fan_args=None, trainable=None,
                 raw_patch_size=128, loss_metric='L2'):
        """
        Setup the model of the entire acquisition and distribution workflow with photo manipulation classification:

        raw -> (nip) -> rgb -> (N manipulations) -> [(downsample) ->] (compression) -> (forensics) -> manipulation classes' probabilities

        :param nip_model: '<nip class>[:dirname]' name of the NIP class + dirname to load a pretrained model
        :param manipulations: list of included manipulations ['<manipulation>[:strength]'], e.g., ['sharpen:1', 'resample', 'gaussian:3', 'jpeg:85']
        :param distribution: definition of the distribution channel (set to None for the default down+jpeg(50))
            {
                'downsampling'        : '<method>[:factor]' - e.g., 'pool:2'
                'compression'         : {'jpeg', 'dcn'},
                'compression_params'  : dict with (codec, quality) for jpeg and (dirname) for dcn
            }
        :param trainable: iterable with components that should be fine-tuned, e.g., {'nip'}. The FAN is always fine-tuned.
        :param raw_patch_size: patch size for manipulation training (raw patch - rgb patches may be bigger)
        :param loss_metric: NIP loss metric: L2, L1 or SSIM
        """
        # Sanitize inputs
        if raw_patch_size < 16 or raw_patch_size > 512:
            raise ValueError('The patch size ({}) looks incorrect, typical values should be >= 16 and <= 512'.format(raw_patch_size))

        self._trainable = set() if trainable is None else set(trainable)
        self._trainable.add('fan')
        
        # Setup a default distribution channel
        if distribution is None:
            self._distribution = {
                'downsampling': 'pool:2',
                'compression': 'jpeg',
                'compression_params': {
                    'quality': 50,
                    'codec': 'soft'
                }
            }
        else:
            self._distribution = {}
            self._distribution.update(distribution)
                    
        if ':' in nip_model:
            nip_model, nip_pretrained_dirname = nip_model.split(':')
        else:
            logger.warning('NIP model weights not specified - no model was loaded during workflow setup!')
            nip_pretrained_dirname = None
    
        if not issubclass(getattr(pipelines, nip_model), pipelines.NIPModel):
            raise ValueError(f'Invalid NIP model ({nip_model})! Available NIPs: ({pipelines.supported_models})')
            
        if loss_metric not in ['L2', 'L1', 'SSIM']:
            raise ValueError(f'Invalid loss metric ({loss_metric})!')
        
        # The pipeline -------------------------------------------------------------------------------------------------

        self.nip = getattr(pipelines, nip_model)(loss_metric=loss_metric, patch_size=raw_patch_size)

        if nip_pretrained_dirname is not None:
            self.nip.load_model(nip_pretrained_dirname)
            logger.info('Loaded NIP weights from {}'.format(nip_pretrained_dirname))

        # Several paths for post-processing ----------------------------------------------------------------------------
        with tf.name_scope('distribution'):

            # Parse manipulation specs
            manipulations = manipulations or ['sharpen', 'resample', 'gaussian', 'jpeg']

            self._strengths = {'sharpen': 1, 'resample': 50, 'gaussian': 0.83, 'jpeg': 80, 'awgn': 5.1, 'gamma': 3, 'median': 3}
            
            self._strengths_range = {
                'sharpen': (0.25, 1.5), 
                'resample': (40, 90), 
                'gaussian': (0.5, 7), 
                'jpeg': (50, 90),
                'awgn': (1, 5), 
                'gamma': (1, 5), 
                'median': (3, 9)
            }

            manipulations_set = set()
            for m in manipulations:
                spec = m.split(':')
                manipulations_set.add(spec[0])
                if len(spec) > 1:
                    self._strengths[spec[0]] = float(spec[-1])

            if any(x not in self._strengths.keys() for x in manipulations_set):
                raise ValueError('Unsupported manipulation requested! Available: {}'.format(self._strengths.keys()))

            self._operations = OrderedDict()
            self._forensics_classes = ['native']

            if 'sharpen' in manipulations_set:
                self._operations['sharpen'] = lambda x, strength: tf_helpers.manipulation_sharpen(x, strength, hsv=True)
                self._forensics_classes.append('sharpen:{}'.format(self._strengths['sharpen']))

            if 'resample' in manipulations_set:
                self._operations['resample'] = lambda x, strength: tf_helpers.manipulation_resample(x, strength)
                self._forensics_classes.append('resample:{}'.format(self._strengths['resample']))

            if 'gaussian' in manipulations_set:
                self._operations['gaussian'] = lambda x, strength: tf_helpers.manipulation_gaussian(x, 5, strength)
                self._forensics_classes.append('gaussian:{}'.format(self._strengths['gaussian']))

            if 'jpeg' in manipulations_set:
                self._operations['jpeg'] = jpeg.differentiable_jpeg
                self._forensics_classes.append('jpeg:{}'.format(self._strengths['jpeg']))

            if 'awgn' in manipulations_set:
                self._operations['awgn'] = lambda x, strength: tf_helpers.manipulation_awgn(x, strength / 255)
                self._forensics_classes.append('awgn:{}'.format(self._strengths['awgn']))

            if 'gamma' in manipulations_set:
                self._operations['gamma'] = lambda x, strength: tf_helpers.manipulation_gamma(x, strength)
                self._forensics_classes.append('gamma:{}'.format(self._strengths['gamma']))

            if 'median' in manipulations_set:
                self._operations['median'] = lambda x, strength: tf_helpers.manipulation_median(x, strength)
                self._forensics_classes.append('median:{}'.format(self._strengths['median']))

            assert len(self._forensics_classes) == self.n_classes

        # Configure compression
        if distribution['compression'] == 'jpeg':
            self.codec = jpeg.JPEG(**distribution['compression_params'])
        elif distribution['compression'] == 'dcn':
            self.codec = codec.restore(distribution['compression_params']['dirname'])
        
        if 'dcn' in trainable and len(self.codec.parameters) == 0:
            raise ValueError('The current codec does not appear to be trainable: {}!'.format(self.codec.class_name))

        # Add forensic analysis
        fan_input_patch = 2 * raw_patch_size // self.downsampling_factor
        self.fan = forensics.FAN(n_classes=self.n_classes, patch_size=fan_input_patch, **fan_args)

        # List parameters that need to be optimized
        self._parameters = []
        self._parameters.extend(self.fan.parameters)
        if 'nip' in trainable:
            self._parameters.extend(self.nip.parameters)
        if 'dcn' in trainable:
            self._parameters.extend(self.codec.parameters)            

        self._optimizer = tf.keras.optimizers.Adam()

    @property
    def n_classes(self):
        return len(self._operations) + 1

    def run_workflow(self, batch_x, augment=False, training=False):
        """
        Runs the entire workflow from RAW images to FAN predictions:

        raw --> [isp] -> [manipulations] -> [downsample] -> [compression] -> [fan] -> class probabilities

        Returns: batch_Y, batch_c, batch_C, entropy, probabilities
        """
        batch_Y = self.nip.process(batch_x)
        batch_m = self.run_manipulations(batch_Y, augment)
        batch_c = self.run_downsampling(batch_m)
        batch_C, entropy = self.run_compression(batch_c, True)
        probabilities = self.fan.process(batch_C)

        return batch_Y, batch_c, batch_C, entropy, probabilities

    def run_workflow_to_decisions(self, batch_x):
        prob = self.run_workflow(batch_x)[-1]
        return prob.numpy().argmax(axis=1)

    def run_rgb_to_fan(self, batch_Y):
        batch_m = self.run_manipulations(batch_Y)
        batch_c = self.run_downsampling(batch_m)
        batch_C = self.run_compression(batch_c)
        if not isinstance(batch_C, np.ndarray):
            batch_C = batch_C.numpy()
        return batch_C

    def run_rgb_to_probabilities(self, batch_Y):
        batch_m = self.run_manipulations(batch_Y)
        batch_c = self.run_downsampling(batch_m)
        batch_C = self.run_compression(batch_c)
        probabilities = self.fan.process(batch_C)
        if not isinstance(probabilities, np.ndarray) and hasattr(probabilities, 'numpy'):
            probabilities = probabilities.numpy()
        return probabilities

    def run_manipulations(self, batch_y, randomize=False, override=None):
        y_list = [batch_y]

        override = override if override is not None else self._strengths

        for name, op in self._operations.items():
            s = override[name] if not randomize else np.random.uniform(*self._strengths_range[name])
            y_list.append(op(batch_y, s))

        return tf.concat(y_list, axis=0)

    def manipulations_timing(self, batch_y):
        from datetime import datetime
        y_list = [batch_y]
        times = {}

        for name, op in self._operations.items():
            d1 = datetime.now()
            y_list.append(op(batch_y, self._strengths[name]))
            times[name] = (datetime.now() - d1).total_seconds()

        return times

    @property
    def downsampling_factor(self):
        if self._distribution['downsampling'] == 'none':
            return 1
        elif ':' in self._distribution['downsampling']:
            return int(self._distribution['downsampling'].split(':')[-1])
        else:
            return 2

    def run_downsampling(self, batch_y):
        factor = self.downsampling_factor

        if self._distribution['downsampling'].startswith('pool'):
            imb_down = tf.nn.avg_pool(batch_y, [1, factor, factor, 1], [1, factor, factor, 1], 'SAME')

        elif self._distribution['downsampling'] == 'bilinear':
            imb_down = tf.image.resize(batch_y, [tf.shape(batch_y)[1] // factor, tf.shape(batch_y)[1] // factor], method='bilinear')

        elif self._distribution['downsampling'] == 'none':
            imb_down = batch_y
        else:
            raise ValueError('Unsupported channel down-sampling {}'.format(self._distribution['downsampling']))
            
        return imb_down

    def run_compression(self, batch_y, return_entropy=False):
        if self._distribution['compression'] == 'jpeg':
            return self.codec.process(batch_y, return_entropy=return_entropy)
        elif self._distribution['compression'] == 'dcn':
            return self.codec.process(batch_y, return_entropy=return_entropy)
        elif self._distribution['compression'] == 'none':
            return batch_y
        else:
            raise ValueError('Unsupported channel compression {}'.format(self._distribution['compression']))

    def _batch_labels(self, batch_size):
        return np.concatenate([x * np.ones((batch_size,), dtype=np.int32) for x in range(self.n_classes)])

    def training_step(self, batch_x, batch_y, lambda_nip=0, lambda_dcn=0, augment=False, learning_rate=1e-4):
        batch_size = batch_x.shape[0]
        with tf.GradientTape() as tape:

            batch_Y, batch_c, batch_C, entropy, probabilities = self.run_workflow(batch_x, augment, training=True)

            # Compute the loss
            loss_ce = self.fan.loss(self._batch_labels(batch_size), probabilities)            
            loss_nip = self.nip.loss(batch_y, batch_Y)
            loss_dcn = self.codec.loss(batch_c, batch_C, entropy)

            loss = loss_ce
            
            if 'nip' in self._trainable:
                loss += lambda_nip * loss_nip
            
            if 'dcn' in self._trainable:
                loss += lambda_dcn * loss_dcn

        self._optimizer.lr.assign(learning_rate)
        grads = tape.gradient(loss, self._parameters)
        if any(np.sum(np.isnan(x)) > 0 for x in grads):
            raise RuntimeError('∇ NaNs: {}'.format({p.name: np.mean(np.isnan(x)) for x, p in zip(grads, self._parameters)}))
        self._optimizer.apply_gradients(zip(grads, self._parameters))
        
        return loss, {'ce': loss_ce, 'nip': loss_nip, 'dcn': loss_dcn}

    def summary_compact(self):
        return '{class_name}[{trainables}]: {nip} -> [{manips}] {pool}{codec}-> {fan}'.format(
            class_name=type(self).__name__,
            nip=self.nip.class_name,
            manips=''.join([x[0] for x in self._forensics_classes]),
            trainables=''.join([x[0] for x in self.trainable_models]),
            pool='' if self._distribution['downsampling'] == 'none' else '-> {} '.format(
                self._distribution['downsampling']),
            codec='' if self.codec is None else '-> {} '.format(self.codec.summary_compact()),
            fan='FAN'
        )

    def summary(self):
        return '{class_name}[opt={trainables}]: {input} -> {nip} -> {n_ops} manipulations [{manips}] {pool}{codec}-> {fan}'.format(
            class_name=type(self).__name__,
            input='(rgb)' if self.nip.x.shape[-1] == 3 else '(raw)',
            nip=self.nip.class_name,
            n_ops=self.n_classes - 1,
            manips=''.join([x[0] for x in self._forensics_classes]),
            trainables=''.join([x[0] for x in self.trainable_models]),
            pool='' if self._distribution['downsampling'] == 'none' else '-> {} '.format(self._distribution['downsampling']),
            codec='' if self.codec is None else '-> {} '.format(self.codec.summary_compact()),
            fan=f'FAN -> (prob. {self.n_classes} classes)'
        )

    def details(self):
        out = [self.summary()]
        out.append('Input         : {} {}'.format(self.nip.x.shape, '(rgb)' if self.nip.x.shape[-1] == 3 else '(raw)'))
        out.append('Camera ISP    : {}'.format(self.nip.summary()))
        out.append('Manipulations : {} -> {}'.format(self.n_classes, self._forensics_classes))
        out.append('Downsampling  : {}'.format(self._distribution['downsampling']))
        out.append('Codec         : {}'.format('' if self.codec is None else self.codec.summary()))
        out.append('Forensics     : {}'.format(self.fan.summary()))
        out.append('Output        : {}'.format(self.fan.y.shape))
        return '\n'.join(out)

    def is_trainable(self, model):
        return model in self._trainable

    @property
    def trainable_models(self):
        return tuple(x for x in self._trainable)