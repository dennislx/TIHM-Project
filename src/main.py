"""
The record is defined as
    28 sensor data + one date information + 6 label information + one patient

The date information:
    2019-04-01 => 2019-06-14 || 2019-06-15 => 2019-06-30
    75 training dates        || 16 testing dates

The target information:
    ['Blood pressure', 'Agitation', 'Body water', 'Pulse', 'Weight', 'Body temperature']
    躁动| 血压| 体温| 身体水分| 脉搏| 体重
    we have 474 records with >= 1 labels
            34  records with >= 2 labels
            1   record  with >= 3 labels

The patient information:
    56 patients with number of records per each patient:
    3.0 -> 39.0    39.0 -> 54.5    54.5 -> 68.0    68.0 -> 91.0
    -------------  --------------  --------------  --------------
            14              14              13              15
            
The input shapes are:
    we have 2032 records for training
            767  records for testing
with n_days=7 rolling window:
    we have 1690 instances for training
            422  instances for testing
"""
import pandas as pd
import datamodule
import models
import utils
import models.utils as mutils
import argparse
import copy

T = lambda _: pd.to_datetime(_).strftime('%Y-%m-%d')

def run(c: utils.Config):
    all_runs = []
    for alc in c.RUNS:
        NName = alc['name'] + alc.get('extra_name', '')
        MModel = models.get_model(alc['name'])
        RRecorder = copy.deepcopy(utils.Recorder(name=NName))
        all_runs.append([NName, MModel, RRecorder])
    DATA = {
        "ML": datamodule.TIHM(c.DPATH, c.RESULT, **c.ML.Data),
        "DL": datamodule.DIHM(c.DPATH, c.RESULT, **c.DL.Data)
    }
    logger = utils.get_logger(name=c.RESULT, savepath=utils.pjoin(c.RESULT, f'{c.EXP_NAME}.txt'), filemode=c.LOGGING)
    logger.info(f'\n\n\n'+'*'*100)
    for seed in c.SEED:
        mutils.seed_everything()
        logger.info(f"Run Experiment with SEED ({seed}) @ {utils.now()}")        
        for i, alc in enumerate(c.RUNS):
            Name, Model, Recorder = all_runs[i]
            logger.info('*'*100 + f"\nRun Model ({Name}) @ {utils.now()}")
            Framework = alc.get('framework', Model.framework)
            for j, train_ds, test_ds in DATA[Framework].evaluate_split(**c.TEST_SPLIT):
                logger.info(f"\nFold {j}: Train up to {T(train_ds.end)} and Test {T(test_ds.start)} - {T(test_ds.end)}")
                model_path = utils.pjoin(c.RESULT, Name, 'model', f'fold{j}-seed{seed}.job', create_if_not_exist=True)
                hist_path  = utils.pjoin(c.RESULT, Name, 'history', f'fold{j}-seed{seed}.job', create_if_not_exist=True)
                hist_index = {'seed': seed, 'fold': j, 'model': Name}
                if Name in c.SKIP_TRAIN:
                    logger.info(f"\tSkip training: \n\t\tload model from {model_path}\n\t\tload result from {hist_path}")
                    fit_model = Model.restore(model_path)
                    recorder = Recorder.restore(hist_path)
                    train_result, valid_result = recorder.get_result(train=True, valid=True)
                else:
                    logger.info("\tStart training {}".format(', '.join(f'{k}={v}' for k,v in alc.items())))
                    save_intermediate = None
                    if alc.get('save_intermediate', False):
                        save_intermediate = utils.pjoin(c.RESULT, Name, 'history', f'fold{j}-seed{seed}.intermediate')
                    fit_model, train_result, valid_result = utils.cross_validation(model_cls=Model, model_args=alc, model_path=model_path, config=c, data=train_ds, save_intermediate=save_intermediate)
                Recorder.log(train_result, index={'stage': 'train', **hist_index})
                Recorder.log(valid_result, index={'stage': 'valid', **hist_index})

                if Name in c.SKIP_TEST:
                    logger.info(f"\tSkip evaluation: \n\t\tload old result from {hist_path}")
                    recorder = Recorder.restore(hist_path)
                    fit_result = list(recorder.get_result(test=True))[0]
                else:
                    logger.info(f"\tStart evaluating Model ({Name})")
                    fit_result = utils.evaluate(model=fit_model, data=test_ds, return_report=True, return_confidence=True, return_loss=True)
                    recorder = None
                Recorder.log(fit_result, index={'stage': 'test', **hist_index})
                Recorder.save(hist_path, index=hist_index)
                logger.info(f"\nFinish Model ({Name}) @ {utils.now()}")
                logger.info('\n' + Recorder.summary(**hist_index).to_string())
            logger.info("*"*100 + '\n\n')
    utils.summarize(all_runs, logname=c.RESULT, savepath=utils.pjoin(c.RESULT, f'{c.EXP_NAME}_end_results.csv'), **c.REPORT)
            
############################################
## Parse Argument
############################################
def parse_args():
    parser = argparse.ArgumentParser( description="TIHM Experiment" )
    parser.add_argument( '-c', '--config', default=None, help=( "initial configuration YAML file for our experiment's setup") )
    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    run(utils.Config(args.config))