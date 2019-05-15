from __future__ import division
import os
from nose.tools import assert_equal
from functools import partial
from logbook import Logger
import mxnet as mx
import numpy as np
import scipy.io as sio
#import scipy.stats as sstat
from itertools import product
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
#from lru import LRU
from . import Dataset as Base
from .. import Combo, Trial
from ... import utils, constant
import types
from ...genIndex import genIndex
from SINGLESRC_rawemg_feature_singlestream_iter import RawDataFeatureImageSingleStreamIter
from MULTISRC_rawemg_feature_multistream_iter import RawDataFeatureImageMultiSourceInputIter
from SINGLESRC_feature_chwise_multistream_iter import ChwiseFeatureMultiStreamIter
from SINGLESRC_feature_chwise_multistream_deepfusion_iter import ChwiseFeatureMultiStreamIter_deepfusion
from MULTISRC_feature_chwise_multistream_iter import ChwiseFeatureMultiStreamIter_v2
from MULTISRC_rawemg_feature_chwise_multistream_iter import ChwiseRawIMGFeatureMultiStreamIter
from MULTISRC_rawemg_feature_chwise_multistream_iter_v2 import ChwiseRawIMGFeatureMultiStreamIter_v2
from MULTISRC_chwise_feature_sigimg_iter import ChwiseFeature_SigImg_Iter
from MULTISRC_chwise_feature_sigimgv2_iter import ChwiseFeature_SigImgv2_Iter
from SINGLESRC_featuresigimg_iter import FeatureSigImgData
from SINGLESRC_featuresigimgv2_iter import FeatureSigImgData_v2
from SINGLESRC_featuresigimgv2_deepfusion_iter import FeatureSigImgData_v2_deepfusion
from SINGLESRC_feature_chwise_multistream_featuresigimgv2_iter import ChwiseFeature_FeatureSigImgv2_Iter
from SINGLESRC_feature_chwise_multistream_2streamfeaturesigimgv2_iter import ChwiseFeature_TwoStreamFeatureSigImgv2_Iter
from MULTISRC_rawemg_feature_sigimgv2_iter import Feature_RawSEMG_Multistream_SigImg_Iter
#from featuremap_iter import FeatureMapData
#from featureimage_iter import FeatureImageData
#from chdiffimage_iter import ChDiffImageData
#from chdiffsigimage_iter import ChDiffSigImageData
#from chdiff_sigimg_multistream_iter import ChDiffSigImgMultiStreamData
#from featureimg_rawimg_multistream_iter import FeatureRawImgMultiStreamData
#from ch_multistream_iter import ChMultiStreamData
#from sigimg_rawimg_multistream_iter import SigImgRawImgMultiStreamData
#from frame_multistream_iter import FrameMultiStreamData
#from simplestacked_iter import SimpleStackedData

logger = Logger(__name__)

NUM_SEMG_ROW = 1
NUM_SEMG_COL = 16
FRAMERATE = 200
PREPROCESS_KARGS = dict(
    framerate=FRAMERATE,
    num_semg_row=NUM_SEMG_ROW,
    num_semg_col=NUM_SEMG_COL
)
WINDOW = 40
STRIDE = 20

class Dataset(Base):

    name = 'ninapro-db1-rawdata-semgfeature-multisource'
   
#    img_preprocess = constant.ACTIVITY_IMAGE_PREPROCESS
#    feature_extraction_winlength = constant.FEATURE_EXTRACTION_WIN_LEN
#    feature_extraction_winstride = constant.FEATURE_EXTRACTION_WIN_STRIDE   
    num_semg_row = 1
    num_semg_col = 16
    
    feature_extraction_winlength = constant.FEATURE_EXTRACTION_WIN_LEN
    feature_extraction_winstride = constant.FEATURE_EXTRACTION_WIN_STRIDE
    
    feature_names = constant.FEATURE_LIST
   
    semg_root = '/home/weiwentao/public-2/wwt/ninapro-db5' 
 #   semg_root = None


    feature_root = ('/home/weiwentao/public-2/wwt/ninapro-feature/ninapro-db5-var-raw-flip-prepro-lowpass-win-%d-stride-%d/' % (feature_extraction_winlength, 
                                                                                                                          feature_extraction_winstride))             
       
    subjects = list(range(10))
    gestures = list(range(13, 53))
    gestures.append(0)
    trials = list(range(6))

#    def __init__(self, root):
#        self.root = root

#    @classmethod
#    def get_preprocess_kargs(cls):
#        return dict(
#            framerate=cls.framerate,
#            num_semg_row=cls.num_semg_row,
#            num_semg_col=cls.num_semg_col
#        )

    def get_trial_func(self, *args, **kargs):
        return GetTrial(*args, **kargs)

    def get_dataiter(self, get_trial, combos, feature_name, window=1, adabn=False, mean=None, scale=None, **kargs):
        
        print ("Use %s, semg row = %d semg col = %d window=%d" % (feature_name,
                                                                  self.num_semg_row,
                                                                  self.num_semg_col,
                                                                  window)) 
#        print self.gestures                                                          
                                                                  
        def data_scale(data):
            if mean is not None:
                data = data - mean
            if scale is not None:
                data = data * scale
            return data 
        
       
        combos = list(combos)
                
        data = []
        feature = []
        gesture = []
        subject = []
        segment = []            
  
        if self.semg_root is None:
            for combo in combos:
                trial = get_trial(self.semg_root, self.feature_root, combo=combo)
                feature.append(data_scale(trial.data[0]))
                gesture.append(trial.gesture)
                subject.append(trial.subject)
                segment.append(np.repeat(len(segment), len(feature[-1])))  
                logger.debug('MAT loaded') 
                data = feature
        else:    
            for combo in combos:
                trial = get_trial(self.semg_root, self.feature_root, combo=combo)
                data.append(data_scale(trial.data[0]))
                feature.append(data_scale(trial.data[1]))
                gesture.append(trial.gesture)
                subject.append(trial.subject)
                segment.append(np.repeat(len(segment), len(data[-1])))  
                logger.debug('MAT loaded')   
        
            
        
        if not data:
           logger.warn('Empty data')
           return
           
        
        index = []
        n = 0
        for seg in data:
            index.append(np.arange(n, n + len(seg) - window + 1))
#            index.append(np.arange(n, n + len(seg)))
            n += len(seg)
        index = np.hstack(index)
        logger.debug('Index made')        
        logger.debug('Segments: {}', len(data))        
         
        logger.debug('First segment data shape: {}', data[0].shape)
#        data = np.vstack(data).reshape(-1, 1, self.num_semg_row, self.num_semg_col)
        data = np.vstack(data)
        if len(data.shape) == 2:
          print 'Using 2D data'
          data = data.reshape(data.shape[0], 1, 1, -1)
        else:
          data = data.reshape(data.shape[0], 1, -1, self.num_semg_row*self.num_semg_col)
        logger.debug('Reshaped data shape: {}', data.shape)
        print 'data shape:', data.shape

        logger.debug('First segment feature shape: {}', feature[0].shape)
        feature = np.vstack(feature)
        if len(feature.shape) == 2:
          print 'Using 2D features'
          feature = feature.reshape(data.shape[0], 1, 1, -1)
        else:
          feature = feature.reshape(feature.shape[0], 1, -1, self.num_semg_row*self.num_semg_col)       
        logger.debug('Reshaped feature shape: {}', feature.shape)
        
        logger.debug('Data and feature stacked')
        
        gesture = get_index(np.hstack(gesture))
        subject = get_index(np.hstack(subject))
        segment = np.hstack(segment)
        
        label = []
        

        label.append(('gesture_softmax_label', gesture))
        

        
        logger.debug('Make data iter')
        
#        assert (feature_name == 'rawfeatureimg' or feature_name == 'rawsemg_feature_multisource' or feature_name == 'rawsemg_feature_singlestream')    
        
        if feature_name == 'rawfeatureimg': 
                return RawFeatureImageData(
                        data=OrderedDict([('data', data)]),
                        label=OrderedDict(label),
                        gesture=gesture.copy(),
                        subject=subject.copy(),
                        segment=segment.copy(),
                        index=index,
                        adabn=adabn,
                        window=window,
                        num_gesture=gesture.max() + 1,
                        num_subject=subject.max() + 1,
                        **kargs
                )        
        elif feature_name == 'rawsemg_feature_singlestream':
           assert self.semg_root is not None
           data = np.concatenate((data, feature),axis=2) 
#           data = data.reshape(data.shape[0], -1);
#           print data.shape
#           data = StandardScaler().fit_transform(data)
#           data = data.reshape(data.shape[0], 1, -1, self.num_semg_row*self.num_semg_col)           
           
           return RawDataFeatureImageSingleStreamIter(
                                        data=OrderedDict([('data', data)]),
                                        label=OrderedDict(label),
                                        gesture=gesture.copy(),
                                        subject=subject.copy(),
                                        segment=segment.copy(),
                                        index=index,
                                        adabn=adabn,
                                        window=window,
                                        num_gesture=gesture.max() + 1,
                                        num_subject=subject.max() + 1,
                                        **kargs
                                      )          
        elif feature_name == 'rawsemg_feature_multisource':
           assert self.semg_root is not None 
           return RawDataFeatureImageMultiSourceInputIter(
                                        data=OrderedDict([('semg', data), ('feature', feature)]),
                                        label=OrderedDict(label),
                                        gesture=gesture.copy(),
                                        subject=subject.copy(),
                                        segment=segment.copy(),
                                        index=index,
                                        adabn=adabn,
                                        window=window,
                                        num_gesture=gesture.max() + 1,
                                        num_subject=subject.max() + 1,
                                        **kargs
                                      )   
        elif feature_name == 'chwise_feature_multistream':
            assert self.semg_root is None
            return ChwiseFeatureMultiStreamIter(
                                        data=OrderedDict([('data', data)]),
                                        label=OrderedDict(label),
                                        gesture=gesture.copy(),
                                        subject=subject.copy(),
                                        segment=segment.copy(),
                                        index=index,
                                        adabn=adabn,
                                        window=window,
                                        num_gesture=gesture.max() + 1,
                                        num_subject=subject.max() + 1,
                                        **kargs
                                      )
        
        elif feature_name == 'chwise_feature_multistream_deepfusion':
            assert self.semg_root is None
            return ChwiseFeatureMultiStreamIter_deepfusion(
                                        data=OrderedDict([('data', data)]),
                                        label=OrderedDict(label),
                                        gesture=gesture.copy(),
                                        subject=subject.copy(),
                                        segment=segment.copy(),
                                        index=index,
                                        adabn=adabn,
                                        window=window,
                                        num_gesture=gesture.max() + 1,
                                        num_subject=subject.max() + 1,
                                        **kargs
                                      )                              
                                      
        elif feature_name == 'chwise_feature_multistream_v2':
            assert self.semg_root is None
            return ChwiseFeatureMultiStreamIter_v2(
                                        data=OrderedDict([('data', data)]),
                                        label=OrderedDict(label),
                                        gesture=gesture.copy(),
                                        subject=subject.copy(),
                                        segment=segment.copy(),
                                        index=index,
                                        adabn=adabn,
                                        window=window,
                                        num_gesture=gesture.max() + 1,
                                        num_subject=subject.max() + 1,
                                        **kargs
                                      )       
                                      
                                      
                                      
        elif feature_name == 'chwise_feature_rawimg_multistream':
            assert self.semg_root is not None
            assert self.feature_root is not None
            return ChwiseRawIMGFeatureMultiStreamIter(
                                        data=OrderedDict([('semg', data), ('feature', feature)]),
                                        label=OrderedDict(label),
                                        gesture=gesture.copy(),
                                        subject=subject.copy(),
                                        segment=segment.copy(),
                                        index=index,
                                        adabn=adabn,
                                        window=window,
                                        num_gesture=gesture.max() + 1,
                                        num_subject=subject.max() + 1,
                                        **kargs
                                      )                               
                                                                 
        elif feature_name == 'chwise_feature_rawimg_multistream_v2':
            assert self.semg_root is not None
            assert self.feature_root is not None
            return ChwiseRawIMGFeatureMultiStreamIter_v2(
                                        data=OrderedDict([('semg', data), ('feature', feature)]),
                                        label=OrderedDict(label),
                                        gesture=gesture.copy(),
                                        subject=subject.copy(),
                                        segment=segment.copy(),
                                        index=index,
                                        adabn=adabn,
                                        window=window,
                                        num_gesture=gesture.max() + 1,
                                        num_subject=subject.max() + 1,
                                        **kargs
                                      )                                    
        
        elif feature_name == 'chwise_feature_featuresigimgv2_multistream':
            assert self.semg_root is None
            return ChwiseFeature_FeatureSigImgv2_Iter(
                                        data=OrderedDict([('data', data)]),
                                        label=OrderedDict(label),
                                        gesture=gesture.copy(),
                                        subject=subject.copy(),
                                        segment=segment.copy(),
                                        index=index,
                                        adabn=adabn,
                                        window=window,
                                        num_gesture=gesture.max() + 1,
                                        num_subject=subject.max() + 1,
                                        **kargs
                                      )           
        elif feature_name == 'chwise_feature_2streamfeaturesigimgv2_multistream':
            assert self.semg_root is None
            return ChwiseFeature_TwoStreamFeatureSigImgv2_Iter(
                                        data=OrderedDict([('data', data)]),
                                        label=OrderedDict(label),
                                        gesture=gesture.copy(),
                                        subject=subject.copy(),
                                        segment=segment.copy(),
                                        index=index,
                                        adabn=adabn,
                                        window=window,
                                        num_gesture=gesture.max() + 1,
                                        num_subject=subject.max() + 1,
                                        **kargs
                                      )              
                      
        elif feature_name == 'sigimg_chwise_feature':
            assert self.semg_root is not None
            assert self.feature_root is not None
            return ChwiseFeature_SigImg_Iter(
                                        data=OrderedDict([('semg', data), ('feature', feature)]),
                                        label=OrderedDict(label),
                                        gesture=gesture.copy(),
                                        subject=subject.copy(),
                                        segment=segment.copy(),
                                        index=index,
                                        adabn=adabn,
                                        window=window,
                                        num_gesture=gesture.max() + 1,
                                        num_subject=subject.max() + 1,
                                        **kargs
                                      ) 
        elif feature_name == 'sigimgv2_chwise_feature':
            assert self.semg_root is not None
            assert self.feature_root is not None
            return ChwiseFeature_SigImgv2_Iter(
                                        data=OrderedDict([('semg', data), ('feature', feature)]),
                                        label=OrderedDict(label),
                                        gesture=gesture.copy(),
                                        subject=subject.copy(),
                                        segment=segment.copy(),
                                        index=index,
                                        adabn=adabn,
                                        window=window,
                                        num_gesture=gesture.max() + 1,
                                        num_subject=subject.max() + 1,
                                        **kargs
                                      ) 
        elif feature_name == 'featuresigimg':
            assert self.semg_root is None
            return FeatureSigImgData(
                        data=OrderedDict([('data', data)]),
                        label=OrderedDict(label),
                        gesture=gesture.copy(),
                        subject=subject.copy(),
                        segment=segment.copy(),
                        index=index,
                        adabn=adabn,
                        window=window,
                        num_gesture=gesture.max() + 1,
                        num_subject=subject.max() + 1,
                        **kargs
                )  

        elif feature_name == 'featuresigimg_v2':
            assert self.semg_root is None
            return FeatureSigImgData_v2(
                        data=OrderedDict([('data', data)]),
                        label=OrderedDict(label),
                        gesture=gesture.copy(),
                        subject=subject.copy(),
                        segment=segment.copy(),
                        index=index,
                        adabn=adabn,
                        window=window,
                        num_gesture=gesture.max() + 1,
                        num_subject=subject.max() + 1,
                        **kargs
                ) 

        elif feature_name == 'featuresigimg_v2_deepfusion':
            assert self.semg_root is None
            return FeatureSigImgData_v2_deepfusion(
                        data=OrderedDict([('data', data)]),
                        label=OrderedDict(label),
                        gesture=gesture.copy(),
                        subject=subject.copy(),
                        segment=segment.copy(),
                        index=index,
                        adabn=adabn,
                        window=window,
                        num_gesture=gesture.max() + 1,
                        num_subject=subject.max() + 1,
                        **kargs
                )   
        elif feature_name == 'rawsemg_feature_multisource_multistream_sigimgv2':
           assert self.semg_root is not None 
           return Feature_RawSEMG_Multistream_SigImg_Iter(
                                        data=OrderedDict([('semg', data), ('feature', feature)]),
                                        label=OrderedDict(label),
                                        gesture=gesture.copy(),
                                        subject=subject.copy(),
                                        segment=segment.copy(),
                                        index=index,
                                        adabn=adabn,
                                        window=window,
                                        num_gesture=gesture.max() + 1,
                                        num_subject=subject.max() + 1,
                                        **kargs
                                      )                                                       
                                      
        
        
    def get_one_fold_intra_subject_trials(self):
        return [0, 2, 3, 5], [1, 4]
        
    def get_one_fold_intra_subject_caputo_trials(self):
        return [i - 1 for i in [1, 3, 4, 5, 9]], [i - 1 for i in [2, 6, 7, 8, 10]]
        
    def get_one_fold_inter_subject_test_data(self, fold, batch_size, preprocess,
                               adabn, minibatch, feature_name, window, num_semg_row, num_semg_col, **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials, preprocess=preprocess, feature_names=self.feature_names)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        train_subjects = [19,20,21,24,25]
        test_subjects = [3,15,17]
        num_train_subjects = len(train_subjects)
        
        train = load(
            combos=self.get_combos(product(train_subjects,
                                           self.gestures, self.trials)),
            adabn=adabn,
            mini_batch_size=batch_size // (num_train_subjects - 1 if minibatch else 1),
            shuffle=True,
            feature_name = feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        val = load(
            combos=self.get_combos(product(test_subjects, self.gestures, self.trials)),
            shuffle=False,
            feature_name = feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        return train, val   
    
    def get_inter_subject_data(self, fold, batch_size, preprocess,
                               adabn, minibatch, feature_name, window, num_semg_row, num_semg_col, **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials, preprocess=preprocess, feature_names=self.feature_names)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        subject = self.subjects[fold]
        train = load(
            combos=self.get_combos(product([i for i in self.subjects if i != subject],
                                           self.gestures, self.trials)),
            adabn=adabn,
            mini_batch_size=batch_size // (self.num_subject - 1 if minibatch else 1),
            shuffle=True,
            feature_name = feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        val = load(
            combos=self.get_combos(product([subject], self.gestures, self.trials)),
            shuffle=False,
            feature_name = feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        return train, val
    
    def get_inter_subject_val(self, fold, batch_size,  feature_name, window, num_semg_row, num_semg_col, preprocess=None, **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials, preprocess=preprocess, feature_names=self.feature_names)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        subject = self.subjects[fold]
        val = load(
            combos=self.get_combos(product([subject], self.gestures, self.trials)),
            shuffle=False,
            feature_name = feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        return val
    
    def get_intra_subject_data(self, fold, batch_size, preprocess,
                               adabn, minibatch, feature_name, window, num_semg_row, num_semg_col, **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials, preprocess=preprocess, feature_names=self.feature_names)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        subject = self.subjects[fold // self.num_trial]
        trial = self.trials[fold % self.num_trial]
        train = load(
            combos=self.get_combos(product([subject], self.gestures,
                                           [i for i in self.trials if i != trial])),
            adabn=adabn,
            mini_batch_size=batch_size // (self.num_subject if minibatch else 1),
            shuffle=True,
            feature_name = feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        val = load(
            combos=self.get_combos(product([subject], self.gestures, [trial])),
            shuffle=False,
            feature_name = feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        return train, val
    
    def get_intra_subject_val(self, fold, batch_size, feature_name, window, num_semg_row, num_semg_col, preprocess=None, **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials, preprocess=preprocess, feature_names=self.feature_names)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        subject = self.subjects[fold // self.num_trial]
        trial = self.trials[fold % self.num_trial]
        val = load(
            combos=self.get_combos(product([subject], self.gestures, [trial])),
            shuffle=False,
            feature_name = feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        return val
    
    def get_universal_intra_subject_data(self, fold, batch_size, preprocess,
                                         adabn, minibatch, feature_name, window, num_semg_row, num_semg_col, **kargs): 
        get_trial = self.get_trial_func(self.gestures, self.trials, preprocess=preprocess, feature_names=self.feature_names)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        trial = self.trials[fold]
        train = load(
            combos=self.get_combos(product(self.subjects, self.gestures,
                                           [i for i in self.trials if i != trial])),
            adabn=adabn,
            mini_batch_size=batch_size // (self.num_subject if minibatch else 1),
            shuffle=True,
            feature_name = feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        val = load(
            combos=self.get_combos(product(self.subjects, self.gestures, [trial])),
            shuffle=False,
            feature_name = feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        return train, val
    
    def get_one_fold_intra_subject_val(self, fold, batch_size, feature_name, window, num_semg_row, num_semg_col, preprocess=None, **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials, preprocess=preprocess, feature_names=self.feature_names)             
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        subject = self.subjects[fold]
        _, val_trials = self.get_one_fold_intra_subject_trials()
        val = load(
            combos=self.get_combos(product([subject], self.gestures,
                                           [i for i in val_trials])),
            shuffle=False,
            feature_name = feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        return val
        
    def get_one_fold_intra_subject_caputo_val(self, fold, batch_size, feature_name, window, num_semg_row, num_semg_col, preprocess=None, **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials, preprocess=preprocess, feature_names=self.feature_names)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        subject = self.subjects[fold]
        _, val_trials = self.get_one_fold_intra_subject_caputo_trials()
        val = load(
            combos=self.get_combos(product([subject], self.gestures,
                                           [i for i in val_trials])),
            shuffle=False,
            feature_name = feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        return val    
        
    
    def get_one_fold_intra_subject_data(self, fold, batch_size, preprocess,
                                        adabn, minibatch, feature_name, window, num_semg_row, num_semg_col, **kargs):
           
        get_trial = self.get_trial_func(self.gestures, self.trials, preprocess=preprocess, feature_names=self.feature_names)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        subject = self.subjects[fold]
        train_trials, val_trials = self.get_one_fold_intra_subject_trials()
        train = load(
            combos=self.get_combos(product([subject], self.gestures,
                                           [i for i in train_trials])),
            adabn=adabn,
            mini_batch_size=batch_size // (self.num_subject if minibatch else 1),
            shuffle=True,
            feature_name = feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        val = load(
            combos=self.get_combos(product([subject], self.gestures,
                                           [i for i in val_trials])),
            shuffle=False,
            feature_name = feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        print 'Data loading finished!'    
        return train, val
      
    def get_one_fold_intra_subject_caputo_data(self, fold, batch_size, preprocess,
                                        adabn, minibatch,  feature_name, window, num_semg_row, num_semg_col, **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials, preprocess=preprocess, feature_names=self.feature_names)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        subject = self.subjects[fold]
        train_trials, val_trials = self.get_one_fold_intra_subject_caputo_trials()
        train = load(
            combos=self.get_combos(product([subject], self.gestures,
                                           [i for i in train_trials])),
            adabn=adabn,
            mini_batch_size=batch_size // (self.num_subject if minibatch else 1),
            shuffle=True,
            feature_name = feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        val = load(
            combos=self.get_combos(product([subject], self.gestures,
                                           [i for i in val_trials])),
            shuffle=False,
            feature_name = feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        print 'Data loading finished!'    
        return train, val  
    
    def get_universal_one_fold_intra_subject_data(self, fold, batch_size, preprocess,
                                                  adabn, minibatch, feature_name, window, num_semg_row, num_semg_col, **kargs):
        assert_equal(fold, 0)
        get_trial = self.get_trial_func(self.gestures, self.trials, preprocess=preprocess, feature_names=self.feature_names)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        train_trials, val_trials = self.get_one_fold_intra_subject_trials()
        train = load(
            combos=self.get_combos(product(self.subjects, self.gestures,
                                           [i for i in train_trials])),
            adabn=adabn,
            mini_batch_size=batch_size // (self.num_subject if minibatch else 1),
            shuffle=True,
            feature_name = feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        val = load(
            combos=self.get_combos(product(self.subjects, self.gestures,
                                           [i for i in val_trials])),
            shuffle=False,
            feature_name = feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        return train, val
        
    def get_universal_one_fold_intra_subject_test_data(self, fold, batch_size, preprocess,
                                                  adabn, minibatch, feature_name, window, num_semg_row, num_semg_col, **kargs):
        assert_equal(fold, 0)
        get_trial = self.get_trial_func(self.gestures, self.trials, preprocess=preprocess, feature_names=self.feature_names)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        train_trials, val_trials = self.get_one_fold_intra_subject_trials()
        subjects_for_test = [2,3,17,22,26]  
        num_subject_for_test = len(subjects_for_test)
        train = load(
            combos=self.get_combos(product(subjects_for_test, self.gestures,
                                           [i for i in train_trials])),
            adabn=adabn,
            mini_batch_size=batch_size // (num_subject_for_test if minibatch else 1),
            shuffle=True,
            feature_name = feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        val = load(
            combos=self.get_combos(product(subjects_for_test, self.gestures,
                                           [i for i in val_trials])),
            shuffle=False,
            feature_name = feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        return train, val    
        
    def get_universal_one_fold_intra_subject_caputo_data(self, fold, batch_size, preprocess,
                                                          adabn, minibatch,  feature_name, window, num_semg_row, num_semg_col, **kargs):
        assert_equal(fold, 0)
        get_trial = self.get_trial_func(self.gestures, self.trials, preprocess=preprocess)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        train_trials, val_trials = self.get_one_fold_intra_subject_caputo_trials()
        train = load(
            combos=self.get_combos(product(self.subjects, self.gestures,
                                           [i for i in train_trials])),
            adabn=adabn,
            mini_batch_size=batch_size // (self.num_subject if minibatch else 1),
            shuffle=True,
            feature_name = feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        val = load(
            combos=self.get_combos(product(self.subjects, self.gestures,
                                           [i for i in val_trials])),
            shuffle=False,
            feature_name = feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        return train, val   
        
class RawFeatureImageData(mx.io.NDArrayIter):

    def __init__(self, *args, **kargs):
        print 'Initialization Data Iter!'
        self.random_shift_vertical = kargs.pop('random_shift_vertical', 0)
        self.random_shift_horizontal = kargs.pop('random_shift_horizontal', 0)
        self.random_shift_fill = kargs.pop('random_shift_fill', constant.RANDOM_SHIFT_FILL)
        self.amplitude_weighting = kargs.pop('amplitude_weighting', False)
        self.amplitude_weighting_sort = kargs.pop('amplitude_weighting_sort', False)
        self.downsample = kargs.pop('downsample', None)
        self.shuffle = kargs.pop('shuffle', False)
        self.adabn = kargs.pop('adabn', False)
        self._gesture = kargs.pop('gesture')
        self._subject = kargs.pop('subject')
        self._segment = kargs.pop('segment')
        self._index_orig = kargs.pop('index')
        self._index = np.copy(self._index_orig)
        self.num_gesture = kargs.pop('num_gesture')
        self.num_subject = kargs.pop('num_subject')
        self.mini_batch_size = kargs.pop('mini_batch_size', kargs.get('batch_size'))
        self.random_state = kargs.pop('random_state', np.random)
        self.balance_gesture = kargs.pop('balance_gesture', 0)
        self.num_channel = 1    
        self.window = kargs.pop('window', 0)
        self.num_semg_col = kargs.pop('num_semg_col', 0)
        self.num_semg_row = kargs.pop('num_semg_row', 0)
        super(RawFeatureImageData, self).__init__(*args, **kargs)

        
#        assert (self.data[0][1].shape[3] == self.num_semg_col*self.num_semg_row)    
        
        self.shape1 = self.data[0][1].shape[2]
        self.shape2 = self.data[0][1].shape[3]        
        self.num_data = len(self._index)   
        self.data_orig = self.data
        assert (self.window == 1)
        self.reset()
        

    @property
    def num_sample(self):
        return self.num_data

    @property
    def gesture(self):
        return self._gesture[self._index]

    @property
    def subject(self):
        return self._subject[self._index]

    @property
    def segment(self):
        return self._segment[self._index]

    @property
    def provide_data(self):
         res = [(k, tuple([self.batch_size, self.num_channel] + list([self.shape1, self.shape2]))) for k, v in self.data]
#         res = [(k, tuple([self.batch_size, self.num_channel] + list([self.window, self.num_semg_col*self.num_semg_row]))) for k, v in self.data]
         return res
        

    def _expand_index(self, index):
        return np.hstack([np.arange(i, i + self.window) for i in index])

    def _reshape_data(self, data):
        return data.reshape(-1, self.window, *data.shape[2:])


    def _get_segments(self, a, index):
        b = mx.nd.empty((len(index), self.window) + a.shape[2:], dtype=a.dtype)  
        for i, j in enumerate(index):           
            b[i] = a[j:j + self.window].reshape(self.window, *a.shape[2:])
        return b

    def _getdata(self, data_source):
        """Load data from underlying arrays, internal use only"""
        assert(self.cursor < self.num_data), "DataIter needs reset."
                  
        if data_source is self.data and self.window > 1:  
            if self.cursor + self.batch_size <= self.num_data:
                #  res = [self._reshape_data(x[1][self._expand_index(self._index[self.cursor:self.cursor+self.batch_size])]) for x in data_source]
                res = [self._get_segments(x[1], self._index[self.cursor:self.cursor+self.batch_size]) for x in data_source]
            else:
                pad = self.batch_size - self.num_data + self.cursor
                res = [(np.concatenate((self._reshape_data(x[1][self._expand_index(self._index[self.cursor:])]),
                                        self._reshape_data(x[1][self._expand_index(self._index[:pad])])), axis=0)) for x in data_source]
        else:
            if self.cursor + self.batch_size <= self.num_data:
                res = [(x[1][self._index[self.cursor:self.cursor+self.batch_size]]) for x in data_source]
            else:
                pad = self.batch_size - self.num_data + self.cursor
                res = [(np.concatenate((x[1][self._index[self.cursor:]], x[1][self._index[:pad]]), axis=0)) for x in data_source] 
                
        if data_source is self.data:       
             new_res = []           
             for a in res:
                 new_res.append(a.asnumpy() if isinstance(a, mx.nd.NDArray) else a)
             res = new_res             
             
#             res = [a.reshape(a.shape[0], a.shape[1], -1) for a in res]
#             res = [self._get_sigimg(a.asnumpy() if isinstance(a, mx.nd.NDArray) else a) for a in res]
             res = [a.reshape(a.shape[0], 1, -1, a.shape[3]) for a in res]
#             print res[0].shape

        res = [a if isinstance(a, mx.nd.NDArray) else mx.nd.array(a) for a in res]
        return res

    def _rand(self, smin, smax, shape):
        return (smax - smin) * self.random_state.rand(*shape) + smin

    def _do_shuffle(self):
        if not self.adabn or len(set(self._subject)) == 1:
            self.random_state.shuffle(self._index)
        else:
            batch_size = self.mini_batch_size
            # batch_size = self.batch_size
            # logger.info('AdaBN shuffle with a mini batch size of {}', batch_size)
            self.random_state.shuffle(self._index)
            subject_shuffled = self._subject[self._index]
            index_batch = []
            for i in sorted(set(self._subject)):
                index = self._index[subject_shuffled == i]
                index = index[:len(index) // batch_size * batch_size]
                index_batch.append(index.reshape(-1, batch_size))
            index_batch = np.vstack(index_batch)
            index = np.arange(len(index_batch))
            self.random_state.shuffle(index)
            self._index = index_batch[index, :].ravel()

            for i in range(0, len(self._subject), batch_size):
                # Make sure that the samples in one batch are from the same subject
                assert np.all(self._subject[self._index[i:i + batch_size - 1]] ==
                              self._subject[self._index[i + 1:i + batch_size]])

            if batch_size != self.batch_size:
                assert self.batch_size % batch_size == 0
                assert (self.batch_size // batch_size) % self.num_subject == 0
                self._index = self._index[:len(self._index) // self.batch_size * self.batch_size].reshape(
                    -1, self.num_subject, batch_size).transpose(0, 2, 1).ravel()

    def reset(self):
        self._reset()
        super(RawFeatureImageData, self).reset()

    def _reset(self):
        self._index = np.copy(self._index_orig)

#        if self.amplitude_weighting:
#            assert np.all(self._index[:-1] < self._index[1:])
#            if not hasattr(self, 'amplitude_weight'):
#                self.amplitude_weight = get_amplitude_weight(
#                    self.data[0][1], self._segment, self.framerate)
#            if self.shuffle:
#                random_state = self.random_state
#            else:
#                random_state = np.random.RandomState(677)
#            self._index = random_state.choice(
#                self._index, len(self._index), p=self.amplitude_weight)
#            if self.amplitude_weighting_sort:
#                logger.debug('Amplitude weighting sort')
#                self._index.sort()

        if self.downsample:
            samples = np.arange(len(self._index))
            np.random.RandomState(667).shuffle(samples)
            assert self.downsample > 0 and self.downsample <= 1
            samples = samples[:int(np.round(len(samples) * self.downsample))]
            assert len(samples) > 0
            self._index = self._index[samples]

        if self.balance_gesture:
            num_sample_per_gesture = int(np.round(self.balance_gesture *
                                                  len(self._index) / self.num_gesture))
            choice = []
            for gesture in set(self.gesture):
                mask = self._gesture[self._index] == gesture
                choice.append(self.random_state.choice(np.where(mask)[0],
                                                       num_sample_per_gesture))
            choice = np.hstack(choice)
            self._index = self._index[choice]

        if self.shuffle:
            self._do_shuffle()

        self.num_data = len(self._index)        




def get_index(a):
    '''Convert label to 0 based index'''
    b = list(set(a))
    return np.array([x if x < 0 else b.index(x) for x in a.ravel()]).reshape(a.shape)

class GetTrial(object):

    def __init__(self, gestures, trials, preprocess=None, feature_names=None):
        self.preprocess = preprocess
        self.memo = {}
        self.gesture_and_trials = list(product(gestures, trials))
        self.feature_names = feature_names

    def get_path(self, root, combo):
        return os.path.join(
            root,
            '{c.subject:03d}',
            '{c.gesture:03d}',
            '{c.subject:03d}_{c.gesture:03d}_{c.trial:03d}.mat').format(c=combo)
            
    def get_feature_path(self, root, combo):
        return os.path.join(
            root,
            '{c.subject:03d}',
            '{c.gesture:03d}',
            '{c.subject:03d}_{c.gesture:03d}_{c.trial:03d}').format(c=combo)        

    def __call__(self, semg_root, feature_root, combo):

        assert (feature_root is not None)       
        
        if semg_root is not None:
            path = self.get_path(semg_root, combo)
            if path not in self.memo:
                logger.debug('Load subject semg data {}', combo.subject)
                paths = [self.get_path(semg_root, Combo(combo.subject, gesture, trial))
                         for gesture, trial in self.gesture_and_trials]
                self.memo.update({path: semg_data for path, semg_data in
                                  zip(paths, _get_data(paths, self.preprocess))})
            semg_data = self.memo[path]
            semg_data = semg_data.copy()

        if feature_root is not None:
            path = self.get_feature_path(feature_root, combo)
            if path not in self.memo:
                logger.debug('Load subject semg feature {}', combo.subject)
                paths = [self.get_feature_path(feature_root, Combo(combo.subject, gesture, trial))
                         for gesture, trial in self.gesture_and_trials]                         
                self.memo.update({path: semg_feature for path, semg_feature in
                                  zip(paths, _get_feature(paths, self.feature_names))})
            semg_feature = self.memo[path]
            semg_feature = semg_feature.copy()

        if semg_root is None:
            gesture = np.repeat(combo.gesture, len(semg_feature))
            subject = np.repeat(combo.subject, len(semg_feature))
            
            data = [semg_feature]
                  
            return Trial(data=data, gesture=gesture, subject=subject)
        else:            
            assert (len(semg_data) == len(semg_feature))        
            
            gesture = np.repeat(combo.gesture, len(semg_data))
            subject = np.repeat(combo.subject, len(semg_data))
            
            data = [semg_data, semg_feature]
                  
            return Trial(data=data, gesture=gesture, subject=subject)
        
        


@utils.cached
def _get_data(paths, preprocess):
    #  return list(Context.parallel(
        #  jb.delayed(_get_data_aux)(path, preprocess) for path in paths))
    return [_get_data_aux(path, preprocess) for path in paths]


def _get_data_aux(path, preprocess):
    data = sio.loadmat(path)['data'].astype(np.float32)
    if preprocess:
        data = preprocess(data, **PREPROCESS_KARGS)
    
    chnum = data.shape[1];     
    data = get_segments(data, window=WINDOW, stride=STRIDE)
    data = data.reshape(-1, WINDOW, chnum)    
        
    return data
        
def _get_feature(paths, feature_names):
    #  return list(Context.parallel(
        #  jb.delayed(_get_data_aux)(path, preprocess) for path in paths))
    return [_get_feature_aux(path, feature_names) for path in paths]


def _get_feature_aux(path, feature_names):
    
    logger.debug('Load {}', path)
    
    data = []
    for i in range(len(feature_names)):
        input_dir = path+'_'+feature_names[i]+'.mat'
        mat = sio.loadmat(input_dir)
        tmp = mat['data'].astype(np.float32)        
        where_are_nan = np.isnan(tmp)  
        where_are_inf = np.isinf(tmp)  
        tmp[where_are_nan] = 0  
        tmp[where_are_inf] = 0 
        data.append(tmp)
     
    data = np.concatenate(data, axis=1)
    return data


def get_segments(data, window, stride):
    return windowed_view(
        data.flat,
        window * data.shape[1],
        (window-stride)* data.shape[1]
    )


def windowed_view(arr, window, overlap):
    from numpy.lib.stride_tricks import as_strided
    arr = np.asarray(arr)
    window_step = window - overlap
    new_shape = arr.shape[:-1] + ((arr.shape[-1] - overlap) // window_step,
                                  window)
    new_strides = (arr.strides[:-1] + (window_step * arr.strides[-1],) +
                   arr.strides[-1:])
    return as_strided(arr, shape=new_shape, strides=new_strides)













